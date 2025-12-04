# 用于rollout Qwen-Omni
import os
import json
import argparse
import torch
import multiprocessing as mp
from multiprocessing import Queue, Process, Manager
from tqdm import tqdm
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, GenerationConfig
from qwen_omni_utils import process_mm_info
import re
import time

def load_processed_indices(output_file):
    """加载已处理问题的索引，实现断点续评"""
    if not os.path.exists(output_file):
        return set()
    processed_indices = set()
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                processed_indices.add(json.loads(line.strip())['index'])
            except (json.JSONDecodeError, KeyError):
                continue
    return processed_indices

def count_words(sentence):
    words = sentence.split()
    return len(words)

SYSTEM_PROMPT_BASE = """You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. 
"""

SYSTEM_PROMPT_think = """You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Your primary goal is to analysis and solve user's question.
    First, identify whether this problem requires thinking. If the problem requires thinking, output the thinking process in <think> </think> and final answer inside <answer> </answer>. 
    If no thinking is required, please output <think></think> to denote the empty thinking process, then output answer in <answer> </answer>. The Assistant is encouraged to use 
    the <answer></answer> tag whenever possible, while ensuring accuracy.
"""

TYPE_TEMPLATE = {
    "multiple choice": "Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
}

def extract_choice(reply: str) -> str:
    """
    从 assistant 回复中提取预测选项。
    优先提取 <answer>X</answer>，如果没有，再从 assistant 生成内容中找 A-D。
    """
    try:
        # 找到 'assistant' 开头
        if "\nassistant" in reply:
            reply = reply.split("\nassistant", 1)[1].strip()

        # 优先匹配 <answer>X</answer>
        m = re.search(r"<answer>\s*([A-Z])\s*</answer>", reply, re.I)
        if m:
            return m.group(1).upper()

        # fallback：找孤立的 A-D 字母
        m = re.search(r"\b([A-Z])\b", reply, re.I)
        if m:
            return m.group(1).upper()
        
        return "?"  # 找不到
    except Exception as e:
        print(f"[extract_choice error]: {e}")
        return "?"

GT_RE = re.compile(r"<answer>\s*([A-Z])\s*</answer>", re.I)

def model_worker(gpu_id, task_queue, result_queue, model_path, cache_dir):
    """模型工作进程，每个进程运行在单独的GPU上"""
    try:
        # 设置当前进程使用的GPU
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
        
        print(f"Process {mp.current_process().name}: Loading model on GPU {gpu_id}...")
        
        # 加载模型 - 使用更稳定的加载方式
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": device},  # 明确指定设备映射
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        model.disable_talker()
        model.eval()
        
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        print(f"Process {mp.current_process().name}: Model loaded successfully on GPU {gpu_id}")
        
        # 持续处理任务
        while True:
            try:
                # 从队列获取任务
                item = task_queue.get(timeout=10)
                if item is None:  # 结束信号
                    break
                
                # 处理任务
                result = process_single_item(item, model, processor, device, gpu_id)
                result_queue.put(result)
                
            except Exception as e:
                print(f"Process {mp.current_process().name} error getting task: {e}")
                break
                
    except Exception as e:
        print(f"Process {mp.current_process().name} encountered error during initialization: {e}")
        result_queue.put({"error": f"GPU {gpu_id}: {e}"})

def process_single_item(item, model, processor, device, gpu_id):
    """处理单个数据项"""
    index = item['problem_id']
    
    # 准备多模态输入
    question = item['problem']
    options_str = "\n".join(item['options'])
    content_ls = []
    
    if item['data_type'] == 'audio':
        content_ls.append({"type": "audio", "audio": item['path']['audio']})
        question_text = question + "\n" + options_str
        prompt = question_text + TYPE_TEMPLATE["multiple choice"]
        content_ls.append({"type": "text", "text": prompt})
    elif item['data_type'] == 'image_audio':
        content_ls.append({"type": "image", "image": item['path']['image']})
        content_ls.append({"type": "audio", "audio": item['path']['audio']})
        question_text = question + "\n" + options_str
        prompt = question_text + TYPE_TEMPLATE["multiple choice"]
        content_ls.append({"type": "text", "text": prompt})
    elif item['data_type'] == 'image':
        content_ls.append({"type": "image", "image": item['path']['image']})
        question_text = question + "\n" + options_str
        prompt = question_text + TYPE_TEMPLATE["multiple choice"]
        content_ls.append({"type": "text", "text": prompt})
    else:
        question_text = question + "\n" + options_str
        prompt = question_text + TYPE_TEMPLATE["multiple choice"]
        content_ls.append({"type": "text", "text": prompt})
    
    # 构建 Qwen 的 conversation 格式
    conversation = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": SYSTEM_PROMPT_think}
            ],
        },
        {
            "role": "user",
            "content": content_ls
        },
    ]
    
    # 准备模型输入
    try:
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audio, images, videos = process_mm_info(conversation, use_audio_in_video=False)

        inputs = processor(
            text=text, 
            audio=audio,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )
        
        # 手动将输入移动到设备
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    except Exception as e:
        print(f"Error processing inputs for index {index} on GPU {gpu_id}: {e}")
        response_text = f"Error: {e}"
        pred = "?"
        is_correct = False
    else:
        # 模型推理
        with torch.no_grad():
            res = model.generate(**inputs, max_new_tokens=2048)
            prompt_length = inputs['input_ids'].size(1)
            res = res[:, prompt_length:]
            response_texts = processor.batch_decode(res, skip_special_tokens=True)[0]
        
        answer = item['solution']
        if '<answer>' in answer:
            gt = GT_RE.search(answer).group(1)
        else:
            gt = answer 
        pred = extract_choice(response_texts)
        is_correct = gt == pred
        response_text = response_texts

    # 返回结果
    result_record = {
        "index": index,
        "data_type": item['data_type'],
        "question": question,
        "options": item['options'],
        "correct_answer": item.get('solution', ''),
        "pred": pred,
        "is_correct": is_correct,
        "pred_res": response_text,
        "difficulty_level": item['difficulty_level'],
        "gpu_id": gpu_id
    }
    
    return result_record

def result_writer(output_file, result_queue, total_tasks, stop_event):
    """结果写入进程"""
    processed_count = 0
    pbar = tqdm(total=total_tasks, desc="Processing items")
    
    with open(output_file, 'a', encoding='utf-8') as f_out:
        while processed_count < total_tasks and not stop_event.is_set():
            try:
                result = result_queue.get(timeout=5)
                if "error" in result:
                    print(f"Error from worker: {result['error']}")
                    continue
                
                f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                f_out.flush()
                processed_count += 1
                pbar.update(1)
                
            except Exception:
                continue
    
    pbar.close()
    print(f"Result writer finished. Processed {processed_count} items.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-Omni model on OmniBench dataset with multi-GPU.")
    parser.add_argument('--model-path', type=str, default="Qwen/Qwen2.5-Omni-7B",
                        help='Path or name of the Qwen2.5-Omni model.')
    parser.add_argument('--input-file', type=str, default="./input.jsonl",
                        help='Path to the OmniBench dataset file (JSONL format).')
    parser.add_argument('--output-file', type=str, default="./qwen_omnibench_results.jsonl",
                        help='Path to save the inference results.')
    parser.add_argument('--mm-data-dir', type=str, default="./mm_data",
                        help='Directory where audio and image folders are located.')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Directory to cache the downloaded model.')
    parser.add_argument('--num-gpus', type=int, default=8,
                        help='Number of GPUs to use for inference.')
    args = parser.parse_args()

    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    
    # 加载数据集
    with open(args.input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # 加载已处理的索引
    processed_indices = load_processed_indices(args.output_file)
    print(f"Found {len(processed_indices)} already processed samples. Resuming...")
    
    # 过滤未处理的数据
    todo_items = [item for item in dataset if item['problem_id'] not in processed_indices]
    print(f"Total {len(todo_items)} items to process.")
    
    if not todo_items:
        print("No items to process. Exiting.")
        return

    # 创建管理器和队列
    with Manager() as manager:
        # 创建任务队列和结果队列
        task_queue = manager.Queue()
        result_queue = manager.Queue()
        stop_event = manager.Event()
        
        # 将任务放入队列
        for item in todo_items:
            task_queue.put(item)
        
        # 创建工作进程
        processes = []
        for gpu_id in range(args.num_gpus):
            p = Process(
                target=model_worker,
                args=(gpu_id, task_queue, result_queue, args.model_path, args.cache_dir),
                name=f"Worker-GPU-{gpu_id}"
            )
            p.start()
            processes.append(p)
            print(f"Started worker process for GPU {gpu_id} (PID: {p.pid})")
            # 给每个进程一些启动时间，避免同时加载模型导致内存冲突
            time.sleep(5)
        
        # 启动结果写入进程
        writer_process = Process(
            target=result_writer,
            args=(args.output_file, result_queue, len(todo_items), stop_event),
            name="Result-Writer"
        )
        writer_process.start()
        
        try:
            # 等待所有任务完成
            writer_process.join()
            
            # 发送结束信号给工作进程
            for _ in range(args.num_gpus):
                task_queue.put(None)
            
            # 等待工作进程结束
            for p in processes:
                p.join(timeout=30)
                if p.is_alive():
                    p.terminate()
            
        except KeyboardInterrupt:
            print("Received interrupt signal. Shutting down...")
            stop_event.set()
            
            # 终止所有进程
            for p in processes:
                if p.is_alive():
                    p.terminate()
            if writer_process.is_alive():
                writer_process.terminate()
        
        print(f"Evaluation finished. Results are saved to {args.output_file}")

if __name__ == '__main__':
    main()