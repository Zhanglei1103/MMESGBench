import os
import re
import json
import argparse
import base64
import fitz
from PIL import Image
from uuid import uuid4
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings, ModelOutput
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from eval.eval_score import eval_score, eval_acc_and_f1, show_results,eval_retrieval
from eval.extract_answer import extract_answer
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor

@dataclass
class CausalLMOutputWithPastForInterCoT(CausalLMOutputWithPast):
    selected_vokens: torch.LongTensor = None


client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

cached_image_list = dict()

def encode_image_to_base64(img):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    tmp = os.path.join('/tmp', str(uuid4()) + '.jpg')
    img.save(tmp)
    with open(tmp, 'rb') as image_file:
        image_data = image_file.read()
    ret = base64.b64encode(image_data).decode('utf-8')
    os.remove(tmp)
    return ret


col_model_name = "vidore/colqwen2-v1.0"
col_model = ColQwen2.from_pretrained(
    col_model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",  
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()
processor = ColQwen2Processor.from_pretrained(col_model_name)


def rag_retrievel(question, images_paths):

    query_input = processor.process_queries([question]).to(col_model.device)
    with torch.no_grad():
        query_embedding = col_model(**query_input)
        scores = []
        for img_path in images_paths:
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                scores.append(float('-inf'))
                continue

            image_input = processor.process_images([pil_img]).to(col_model.device)
            with torch.no_grad():
                image_embedding = col_model(**image_input)
            score_matrix = processor.score_multi_vector(query_embedding, image_embedding)
            score = score_matrix[0][0].item()
            scores.append(score)


        sorted_indices = np.argsort(scores)[::-1]
        top5_0_based = sorted_indices[:5]
        top5_pages = (top5_0_based + 1).tolist()
        top5_image_paths = [images_paths[i] for i in top5_0_based]
        print(f"Top5 page indices (1-based): {top5_pages}")
    return top5_pages, top5_image_paths, scores

def process_sample_gpt(sample, args):

    question = sample["question"]
    doc_name = re.sub(r"\.pdf$", "", sample["doc_id"]).split("/")[-1]
    pdf_path = os.path.join(args.document_path, sample["doc_id"])
    with fitz.open(pdf_path) as pdf:
        num_pages = min(args.max_pages, pdf.page_count)
        images_paths = [f"./tmp/{doc_name}_{index+1}.png" for index in range(num_pages)]
        for index, page in enumerate(pdf[:num_pages]):
            tmp_path = f"./tmp/{doc_name}_{index+1}.png"
            if not os.path.exists(tmp_path):
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                pix = page.get_pixmap(dpi=args.resolution)
                pix.save(tmp_path)
        top5_pages, top5_image_paths, scores = rag_retrievel(question, images_paths)

    content = []
    content.append({
        "type": "text",
        "text": question,
    })
    for img_path in top5_image_paths:
        image = Image.open(img_path)
        encoded_image = encode_image_to_base64(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    messages = [{
        "role": "user",
        "content": content
    }]
    return messages, top5_pages, scores


def process_sample(sample, args, mode="png"):
    if "qwen-vl-max" in args.model_name:
        return process_sample_gpt(sample, args)
    else:
        raise AssertionError()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./datasets/samples.json")
    parser.add_argument("--document_path", type=str, default="./datasets/source_doc")
    parser.add_argument("--model_name", type=str, default="qwen-vl-max")
    parser.add_argument("--max_pages", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--max_try", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--extractor_prompt_path", type=str, default="./eval/prompt_for_answer_extraction.md")
    args = parser.parse_args()

    args.output_path = f'./results/results_{args.model_name}_colpali.json'

    with open(args.extractor_prompt_path) as f:
        prompt = f.read()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)

    total_samples = 0
    top1_hits = 0
    top5_hits = 0
    ndcg_sum = 0.0
    for sample in tqdm(samples):
        if "score" in sample:
            score = sample["score"]
        else:
            messages, pred_top5, scores = process_sample(sample, args)
            sample["pred_page"] = pred_top5 
            gt_page = sample["evidence_pages"]
            sample["scores"] = scores

            total_samples += 1
            try:
                gt_page = int(gt_page)
            except:
                pass

            if pred_top5[0] == gt_page:
                top1_hits += 1

            if gt_page in pred_top5:
                top5_hits += 1
                for i, page in enumerate(pred_top5):
                    if page == gt_page:
                        ndcg = 1.0 / np.log2(i + 2)
                        break
            else:
                ndcg = 0.0
            ndcg_sum += ndcg

            try_cnt = 0
            is_success = False

            while True:
                try:
                    if "qwen-vl-max" in args.model_name:
                        response = client.chat.completions.create(
                            model=args.model_name,
                            messages=messages,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            seed=42
                        )
                        response = response.choices[0].message.content
                    else:
                        pass
                    is_success = True
                except Exception as e:
                    try_cnt += 1
                    response = "Failed"
                    if try_cnt > args.max_try:
                        break
                if is_success or try_cnt > args.max_try:
                    break

            sample["response"] = response
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res
            print(extracted_res)
            try:
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            except Exception as e:
                pred_ans = "Failed to extract"
            score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            sample["pred"] = pred_ans
            sample["score"] = score

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))
        print("Pred top5 pages: {}".format(sample["pred_page"]))
        print("Gt page: {}".format(sample["evidence_pages"]))
        print("--------------------------------------")
        
        with open(args.output_path, 'w') as f:
            json.dump(samples, f)
    

    show_results(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))
    eval_retrieval(samples, show_path=re.sub(r"\.json$", ".txt", args.output_path))