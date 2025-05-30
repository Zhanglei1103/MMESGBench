# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# def init_model(cache_path=None):
#     model_path = cache_path if cache_path and cache_path != "None" else "mistralai/Mixtral-8x7B-Instruct-v0.1"
#     tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,  # 建议用 bfloat16 或 float16
#         device_map="auto",          # 自动分配到 GPU（推荐用于大模型）
#         trust_remote_code=True
#     )
#     model.eval()
#     model.tokenizer = tokenizer
#     return model

    
# def get_response_concat(model, question, context, max_new_tokens=512, temperature=0.7):
#     prompt = f"""You are a helpful assistant. Answer the following question based on the context provided.

# Context:
# {context}

# Question: {question}
# Answer:"""

#     inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device)

#     try:
#         with torch.no_grad():
#             generated_ids = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=(temperature > 0),
#                 temperature=temperature,
#                 top_p=0.9,
#                 repetition_penalty=1.0,
#                 pad_token_id=model.tokenizer.eos_token_id
#             )
#         output = model.tokenizer.decode(generated_ids[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
#     except Exception as e:
#         print(f"[!] Model inference failed: {e}")
#         output = "Failed"
    
#     return output


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def init_model(cache_path=None):
    model_path = cache_path if cache_path and cache_path != "None" else "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    model.tokenizer = tokenizer
    return model

def get_response_concat(model, question, context, max_new_tokens=512, temperature=0.7):
    prompt = f"""You are a helpful assistant. Answer the following question based on the context provided.

Context:
{context}

Question: {question}
Answer:"""

    try:
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.0,
                pad_token_id=model.tokenizer.eos_token_id or model.tokenizer.pad_token_id,
                return_dict_in_generate=False
            )
        output = model.tokenizer.decode(
            generated_ids[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
    except Exception as e:
        print(f"[!] Model inference failed: {e}")
        output = "Failed"

    return output
