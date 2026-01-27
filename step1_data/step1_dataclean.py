#数据集处理
import os
import io
import json
import time
import sys
import re
import unicodedata
import pyarrow.parquet as pq
import traceback
import zstandard as zstd
import pyarrow.parquet as pq
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

class FormatHandler():
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name
        self.seen_texts = set() #初始化去重的set

    def get_file_list(self) -> list:
        file_list = []
        for root, _, files in os.walk(self.input_path):  # 递归遍历所有子目录
            for file in files:
                if file.endswith(".jsonl") or file.endswith(".parquet"):
                    file_list.append(os.path.join(root, file))
        return file_list
    
    def process_one_line(self, line, fout) -> bool:
        raise NotImplementedError
    
    def process_one_file(self, file_path, max_lines=None) -> (int, int):
        line_count = 0
        jump_count = 0
        seen_texts = set() 
    
        file_path_full = file_path
        with open(self.output_path, "a", encoding="utf-8") as fout:
            with open(file_path_full, "r", encoding="utf-8") as fin:
                for line in fin:
                    if max_lines and line_count >= max_lines:
                        break  
                    try:
                        processed_text = self.process_one_line(line, fout) 
                        if not processed_text:
                            jump_count += 1 
                            continue  
    
                        text = processed_text.strip()
                        if text in seen_texts:
                            jump_count += 1 
                            continue  
    
                        seen_texts.add(text)  
                        if fout is not None:
                            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")  # ✅ 保证写入去重文本
                        line_count += 1
                    except Exception as e:
                        print(f"[Error] 处理行失败: {e}")
                        jump_count += 1  # 记录错误行
        return line_count, jump_count
    
    def process_all(self, max_lines=None):
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print("[log][{}] number of files is {:d}".format(self.dataset_name, len(file_list)))
        for file in file_list:
            line_count, jump_count = 0, 0
            try:
                line_count, jump_count = self.process_one_file(file, max_lines=max_lines)
            except Exception as e:
                print("[exception][{}] process file {} failed: {}".format(self.dataset_name, file, e))
                print(traceback.format_exc())
            line_count_all += line_count
            jump_count_all += jump_count
        print("[log][{}] timecost is {:.2f} s!".format(self.dataset_name, time.time() - st))
        print("[log][{}] line_count is {:d}, jump_count is {:d}".format(self.dataset_name, line_count_all, jump_count_all))
        print(f"[log] {self.dataset_name} 数据处理完毕！")  

    def quality_assurance(self, line) -> bool:
        if len(line) < 10:  # 允许短文本，但限制过短的内容
            print("文本太短")
            return False
        if line.count("\n") > 200:  # 提高换行阈值
            print("换行符太多，一共{}".format(line.count("\n")))
            return False
        return True

    def quality_assurance_math(self, line) -> bool:
        if len(line) < 10:  # 允许短文本，但限制过短的内容
            print("文本太短")
            return False
        return True

    def zh_process(self, line) -> str:
        # 0. None 处理成空字符串
        if line is None:
            return ""
        # 1. unicode 统一
        line = unicodedata.normalize("NFKC", line)
        # 2. 替换\n
        # line = line.replace("\n\n", "\n")
        # 3. 移除 Unicode 控制字符（如 \u200b, \ufeff）
        line = "".join(ch for ch in line if unicodedata.category(ch)[0] != "C")
        # 4. 将\r替换成\t
        line = line.replace("\r", "").replace("\t", " ")  # 规范换行和制表符
        line = line.strip()  # 去除前后空格
        return line


class SkypileFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super().__init__(input_path, output_path, dataset_name)
        self.seen_texts = set()
        self.duplicate_count = 0  # 统计重复数量
        self.empty_count = 0  # 统计空白数量 

    def get_file_list(self) -> list:
        file_list = []
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if file.endswith(".jsonl"):
                    file_list.append(os.path.join(root, file))
        return file_list

    def process_one_line(self, line, fout) -> bool:
        try:
            line = line.strip()  # 去掉空格和换行
            data = json.loads(line)  # 确保 JSON 解析成功
            text = data.get("text", "").strip()
    
            # 检查 text 是否为空或是重复
            if not text:
                self.empty_count += 1  # 只统计，不打印
                return False
            if text in self.seen_texts:
                self.duplicate_count += 1  # 只统计，不打印
                return False
    
            # 如果没有跳过，则添加到 seen_texts 并写入
            self.seen_texts.add(text)  # 记录新的文本
            fout.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")  # 直接写入
            return True  # 处理成功
    
        except Exception as e:
            # 减少错误日志输出频率
            return False

    def process_one_file(self, file_path, max_lines=None) -> (int, int):
        """对单个 Skypile 文件进行格式检查并写入 JSONL"""
        line_count = 0
        valid_count = 0
        invalid_count = 0
        
        # 记录本文件开始时的重复和空白计数
        file_start_duplicate = self.duplicate_count
        file_start_empty = self.empty_count
        
        print(f"[START] 开始处理文件: {os.path.basename(file_path)}")

        with open(file_path, "r", encoding="utf-8") as fin, open(self.output_path, "a", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if max_lines and valid_count >= max_lines:  # 限制有效行数
                    break
                
                line_count += 1
                # 每处理10000行打印一次进度
                if line_count % 10000 == 0:
                    file_duplicate = self.duplicate_count - file_start_duplicate
                    file_empty = self.empty_count - file_start_empty
                    print(f"[PROGRESS] 已处理 {line_count} 行，有效 {valid_count} 行，重复 {file_duplicate} 个，空白 {file_empty} 个")
                
                if self.process_one_line(line, fout):  # 现在 `process_one_line` 负责写入
                    valid_count += 1
                else:
                    invalid_count += 1

        # 计算本文件的重复和空白数
        file_duplicate = self.duplicate_count - file_start_duplicate
        file_empty = self.empty_count - file_start_empty
        print(f"[DONE] {os.path.basename(file_path)} - 总行数: {line_count}, 有效: {valid_count}, 重复: {file_duplicate}, 空白: {file_empty}")
        return valid_count, invalid_count
    
    def process_all(self, max_lines=None):
        st = time.time()
        valid_count_all = 0
        invalid_count_all = 0
        file_list = self.get_file_list()
        print(f"[INFO] {self.dataset_name}: 发现 {len(file_list)} 个文件。")
        print("=" * 80)
    
        for idx, file in enumerate(file_list, 1):
            print(f"\n[{idx}/{len(file_list)}] 处理文件...")
            try:
                valid_count, invalid_count = self.process_one_file(file, max_lines=max_lines)
                valid_count_all += valid_count
                invalid_count_all += invalid_count
            except Exception as e:
                print(f"[EXCEPTION] {self.dataset_name}: 处理文件 {file} 失败: {e}")
                print(traceback.format_exc())
    
        print("\n" + "=" * 80)
        print(f"[SUMMARY] {self.dataset_name} 处理完成！")
        print(f"  - 耗时: {time.time() - st:.2f} 秒")
        print(f"  - 有效行数: {valid_count_all}")
        print(f"  - 跳过行数: {invalid_count_all} (重复: {self.duplicate_count}, 空白: {self.empty_count})")
        print(f"  - 去重后文本数: {len(self.seen_texts)}")
        print("=" * 80)

def main_run():
    input_path_root = os.path.dirname(os.path.realpath(__file__))
    output_path_root = input_path_root + "/basic_clean"
    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "skypile": ("/mnt/data/workspace/sy_transformers/deepseekv3/SkyPile/data", SkypileFormatHandler),
    }

    for dataset_name, info in dataset_process_info.items():
        input_path = info[0]
        Handler = info[1]
        output_path = os.path.join(output_path_root, f"processed_{dataset_name}.jsonl")
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()

    
if __name__ == "__main__":
    main_run()
