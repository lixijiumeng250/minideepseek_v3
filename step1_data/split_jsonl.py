import os
import argparse

def split_jsonl_file(input_file, output_folder, max_chunk_size):
    """
    将一个 JSONL 文件拆分为多个小文件，每个文件最大大小为 max_chunk_size（字节）。
    拆分后的文件命名为 {原文件名}_part{序号}.jsonl，并存储在 output_folder 文件夹中。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    part_num = 0
    out_file_path = os.path.join(output_folder, f"{base_name}_part{part_num}.jsonl")
    out_file = open(out_file_path, "w", encoding="utf-8")
    current_size = 0  # 当前输出文件的字节数

    with open(input_file, "r", encoding="utf-8", errors='replace') as f:
        for line in f:
            # 计算该行占用的字节数（utf-8 编码）
            line_size = len(line.encode("utf-8"))
            # 如果加上这一行后超出目标大小，则关闭当前文件，新建一个文件
            if current_size + line_size > max_chunk_size:
                out_file.close()
                part_num += 1
                out_file_path = os.path.join(output_folder, f"{base_name}_part{part_num}.jsonl")
                out_file = open(out_file_path, "w", encoding="utf-8")
                current_size = 0
            out_file.write(line)
            current_size += line_size

    out_file.close()
    print(f"已将 {input_file} 拆分为 {part_num+1} 个部分，存放于文件夹 {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="拆分 JSONL 文件为多个小文件")
    parser.add_argument(
        "--max_size_mb",
        type=int,
        default=300,
        help="每个小文件的最大大小（单位 MB，默认 300MB，可根据需求设为100~500MB）"
    )
    args = parser.parse_args()
    max_chunk_size = args.max_size_mb * 1024 * 1024  # 转换为字节

    # 需要拆分的 JSONL 文件列表
    files = [
        "/mnt/data/workspace/sy_transformers/minideepseek/step1_data/basic_clean/processed_skypile.jsonl"
    ]

    for input_file in files:
        # 以去掉扩展名后的文件名作为文件夹名，例如 processed_ape210k
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_folder = os.path.join(os.path.dirname(input_file), base_name)
        split_jsonl_file(input_file, output_folder, max_chunk_size)

if __name__ == "__main__":
    main()