import os
import json
from unstructured.partition.pdf import partition_pdf
from multiprocessing import Pool
from tqdm import tqdm
from typing import Dict

# 统一的 JSONL 输出文件夹
JSONL_OUTPUT_DIR = "/mnt/data/workspace/sy_transformers/deepseekv3/v3/data/PDF_result"
os.makedirs(JSONL_OUTPUT_DIR, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    使用 unstructured 库提取 PDF 文件的文本内容
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        包含文本内容的字典，格式为 {"text": textcontent}
    """
    try:
        print(f"正在解析PDF文件: {pdf_path}")
        
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # 高精度提取
            extract_images_in_pdf=True,  # 提取pdf中的图片
            extract_image_block_types=["Table", "Image"],  # 提取表格和图片
            languages=["eng", "chi_sim"],  # 支持英文和简体中文
            infer_table_structure=True,  # 推断表格结构
            include_page_breaks=True  # 包含页码信息
        )
        
        # 提取所有文本内容
        text_parts = []
        for element in elements:
            if hasattr(element, 'text') and element.text:
                text_parts.append(element.text)
        
        # 合并所有文本
        text_content = "\n\n".join(text_parts)
        
        return {"text": text_content}

    except Exception as e:
        print(f"处理文件 {pdf_path} 时出错: {e}")
        return {"text": "", "error": str(e)}


def process_single_pdf(pdf_path: str) -> Dict[str, str]:
    """处理单个PDF文件的包装函数（用于多进程）"""
    return extract_text_from_pdf(pdf_path)


def process_folder(pdf_folder: str, num_processes: int = 8):
    """
    处理单个文件夹中的所有 PDF 并保存为 JSONL
    每个PDF文件保存为单独的 pdfProcessN.jsonl 文件
    
    Args:
        pdf_folder: PDF文件夹路径
        num_processes: 并行进程数
    """
    pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"文件夹 {pdf_folder} 中没有PDF文件，跳过")
        return
    
    print(f"正在处理文件夹: {pdf_folder}，发现 {len(pdf_files)} 个 PDF 文件...")
    
    # 使用多进程处理
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_pdf, pdf_files),
            total=len(pdf_files),
            desc=f"处理PDF文件"
        ))
    
    # 保存每个成功处理的结果为单独的文件
    success_count = 0
    failed_count = 0
    
    for result in results:
        # 跳过失败或空文本的结果
        if "error" in result or not result.get("text", "").strip():
            failed_count += 1
            continue
        
        success_count += 1
        # 为每个成功的PDF生成带序号的文件名（序号连续）
        output_file = os.path.join(JSONL_OUTPUT_DIR, f"pdfProcess{success_count}.jsonl")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"结果已保存至: {JSONL_OUTPUT_DIR}")
    print(f"成功处理: {success_count} 个文件")
    if failed_count > 0:
        print(f"跳过失败: {failed_count} 个文件")


def process_single_pdf_file(pdf_path: str, output_path: str = None):
    """
    处理单个PDF文件并保存结果
    
    Args:
        pdf_path: PDF文件路径
        output_path: 输出JSON文件路径
    """
    if not os.path.exists(pdf_path):
        print(f"文件不存在: {pdf_path}")
        return None
    
    result = extract_text_from_pdf(pdf_path)
    
    # 如果处理失败，跳过保存
    if "error" in result:
        print(f"处理失败，跳过保存")
        return None
    
    if output_path is None:
        # 默认输出路径
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(JSONL_OUTPUT_DIR, f"{base_name}.json")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存至: {output_path}")
    
    return result


def main():
    root_folder = "/mnt/data/workspace/sy_transformers/deepseekv3/v3/data"
    process_folder(root_folder, num_processes=3)
    print("所有文件夹处理完成！")


if __name__ == "__main__":
    # 处理整个目录树
    main()
    
    # 处理单个PDF文件
    # pdf_file = "/mnt/data/workspace/sy_transformers/deepseekv3/v3/data/万卷预训练数据集.pdf"
    # process_single_pdf_file(pdf_file)
