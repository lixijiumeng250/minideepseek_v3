import glob
import json
import os
import sys
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Process
import numpy as np
from tqdm import tqdm
import random
import time
from pathlib import Path
from typing import List

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from transformers import AutoTokenizer

random.seed(666)

def process_line_text(text, tokenizer, max_length=131072):
    text_ids = tokenizer.encode(text)

    if any(t >= tokenizer.vocab_size for t in text_ids):
        print(f"token_id 超过 vocab_size）: {max(text_ids)}")
        return []

    text_ids = np.array(text_ids, dtype=np.int32)

    if len(text_ids) > max_length:
        print(f"超长文本{len(text_ids)}!!!")
        return [text_ids[i:i+max_length] for i in range(0, len(text_ids), max_length)]
    else:
        return [text_ids]

def process_jsonl_files(set_name, file_dir_list, builder, tokenizer, chunk_size, max_length=131072):
    cache_tokens = []
    total_tokens = 0 

    for file_dir in file_dir_list:
        print(f"JSONL: {file_dir}")
        with open(file_dir, encoding="utf-8") as f:
            counter = 0
            for line in tqdm(f, desc=f"Reading {file_dir}"):
                try:
                    text = json.loads(line)["text"]
                    #print(text[:100])
                except Exception as e:
                    print(f"file{file_dir}read error: {e}")
                    continue

                text += "<｜end▁of▁sentence｜>"  
                text_id_chunks = process_line_text(text, tokenizer, max_length)

                for chunk in text_id_chunks:
                    cache_tokens.extend(chunk)
                    total_tokens += len(chunk)
                    if total_tokens >= chunk_size:
                        print(f"Adding to builder: {len(cache_tokens[:chunk_size])} tokens")
                        builder.add_array(np.array(cache_tokens[:chunk_size], dtype=np.uint32))
                        print(f"write to bin: {len(cache_tokens[:chunk_size])} tokens")
                        cache_tokens = cache_tokens[chunk_size:] 
                        total_tokens = len(cache_tokens)

    if total_tokens > 0:
        padding_needed = chunk_size - total_tokens
        cache_tokens.extend([tokenizer.pad_token_id] * padding_needed) 
        builder.add_array(np.array(cache_tokens, dtype=np.uint32))

def multiprocess_data(set_name, file_dir_list, destination_path, chunk_size, checkpoint_dir, process_idx=0):
    try:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=f"{set_name}_process{process_idx}", # 避免多个进程写入相同文件
            chunk_size=chunk_size,
            sep_token=tokenizer.pad_token_id,
            dtype=np.int32,
            vocab_size=len(tokenizer),
        )
        process_jsonl_files(set_name, file_dir_list, builder, tokenizer, chunk_size)
        builder.write_reminder()

        print(f"Process {process_idx} process {len(file_dir_list)} files, total time {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"multiprocess_data process {set_name} error: {str(e)}")

def prepare_full(
    source_path: Path,
    checkpoint_dir: Path,
    destination_path: Path,
    chunk_size: int,
    match: str = "",
    max_files: int = 1_000_000_000,
    process_num: int = 64
):
    destination_path.mkdir(parents=True, exist_ok=True)

    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        print(f"\ndataset name: {set_name}")
        t0 = time.time()

        filenames = sorted([Path(p) for p in glob.glob(pattern, recursive=True)])
        if not filenames:
            print(f"no file found for pattern `{pattern}` , skip {set_name}")
            continue

        random.shuffle(filenames)
        filenames = filenames[:max_files]
        print(f"file total: {len(filenames)}")

        actual_process_num = min(process_num, len(filenames))
        file_chunks = np.array_split(filenames, actual_process_num)

        process_list = []
        for process_idx in range(actual_process_num):
            sub_file_list = [str(p) for p in file_chunks[process_idx]]
            print(f"start process {process_idx}, file number: {len(sub_file_list)}")
            process = mp.Process(
                target=multiprocess_data,
                args=(set_name, sub_file_list, destination_path, chunk_size, checkpoint_dir, process_idx)
            )
            process.start()
            process_list.append(process)

        for process in process_list:
            process.join()

        print(f"{set_name} data processing completed, time {time.time() - t0:.2f}s")

filename_sets = {
    "djed_skypile": "/your/path/to/djed_skypile.jsonl",
}

def prepare(
    source_path: Path = Path("/"),
    checkpoint_dir: Path = Path("/your/path/to/tokenizer"),
    destination_path: Path = Path("/your/path/to/final_data"),
    sample: bool = False,
    match: str = "",
    max_files=10000000000,     
    block_size= 2048,              #模型训练的max_len 
    blocks_in_a_chunck= 1024 * 20,  
    process_num=64      
) -> None:
    prepare_fn = prepare_full
    prepare_fn(
        source_path=source_path,
        checkpoint_dir=checkpoint_dir,
        destination_path=destination_path,
        chunk_size=(block_size + 1) * blocks_in_a_chunck,  
        match=match,
        max_files=max_files,
        process_num=process_num  
    )

if __name__ == "__main__":
    import jsonargparse
    from jsonargparse import CLI
    CLI(prepare)
