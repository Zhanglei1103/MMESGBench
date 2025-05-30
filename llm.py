import os
import re
import math
import json
import argparse
import fitz
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm

from eval.eval_score import eval_score, eval_acc_and_f1, show_results
from eval.extract_answer import extract_answer


def load_model(model_name, cache_path):
    if model_name == 'chatglm':
        from models.chatglm import init_model, get_response_concat
    elif model_name == 'Mixtral-Instruct-v0.1':
        from models.Mixtral_Instruct import init_model, get_response_concat
    elif model_name == 'qwen-14b-chat':
        from models.qwen_14b_chat import init_model, get_response_concat
    elif model_name == 'deepseek_llm_chat':
        from models.deepseek_llm_7b_chat import init_model, get_response_concat

    else:
        raise NotImplementedError
    model = init_model(cache_path)
    return model, get_response_concat


def load_md_chunks(md_path, chunk_size=60):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    chunks = ["".join(lines[i:i + chunk_size]) for i in range(0, len(lines), chunk_size)]
    return chunks

def load_questions(args):
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            samples = json.load(f)
    else:
        with open(args.input_path, 'r') as f:
            samples = json.load(f)

    with open(args.extractor_prompt_path) as f:
        prompt = f.read()

    model, get_response_concat = load_model(args.model_name, args.model_cached_path)
    print(type(model))

    for sample in tqdm(samples):
        if "score" in sample:
            score = sample["score"]
        else:
            md_path = os.path.join(args.document_path, sample["doc_id"].replace(".pdf", ".md"))
            chunks = load_md_chunks(md_path, chunk_size=args.chunk_size)

            response = "Failed"
            for chunk in chunks:
                response = get_response_concat(model, sample["question"], chunk, max_new_tokens=args.max_tokens, temperature=args.temperature)
                if response != "Failed":
                    break

            sample["response"] = response
            extracted_res = extract_answer(sample["question"], response, prompt)
            sample["extracted_res"] = extracted_res

            try:
                pred_ans = extracted_res.split("Answer format:")[0].split("Extracted answer:")[1].strip()
                score = eval_score(sample["answer"], pred_ans, sample["answer_format"])
            except:
                pred_ans = "Failed to extract"
                score = 0.0

            sample["pred"] = pred_ans
            sample["score"] = score

        acc, f1 = eval_acc_and_f1(samples)
        print("--------------------------------------")
        print("Question: {}".format(sample["question"]))
        print("Response: {}".format(sample["response"]))
        print("Gt: {}\tPred: {}\tScore: {}".format(sample["answer"], sample["pred"], sample["score"]))
        print("Avg acc: {}".format(acc))
        print("Avg f1: {}".format(f1))

        with open(args.output_path, 'w') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

    show_results(samples, show_path=re.sub("\.json$", ".txt", args.output_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./datasets/samples.json")
    parser.add_argument("--document_path", type=str, default="./datasets/markdowns")
    parser.add_argument("--extractor_prompt_path", type=str, default="./eval/prompt_for_answer_extraction.md")
    parser.add_argument("--model_name", type=str, default="qwen-14b-chat", choices=["chatglm", "Mixtral-Instruct-v0.1","qwen-14b-chat", "deepseek_llm_chat"])
    parser.add_argument("--model_cached_path", type=str, default=None)
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    args = parser.parse_args()
    args.output_path = f'./results/res_{args.model_name}.json'
    args.temperature = 0.1 if args.model_name in ['deepseek_llm_chat'] else 0
    load_questions(args)

