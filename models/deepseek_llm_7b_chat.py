# models/deepseek_llm_chat.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def init_model(cache_path=None):
    model_name = cache_path if (cache_path and cache_path != "None") else "deepseek-ai/deepseek-llm-7b-chat"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    generation_config = GenerationConfig.from_pretrained(model_name)
    generation_config.pad_token_id = generation_config.eos_token_id
    model.tokenizer = tokenizer
    model.generation_config = generation_config
    return model

def get_response_concat(model, question, context, max_new_tokens=1024, temperature=0.7):
    prompt = f"Please answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"

    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        input_tensor = model.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            pad_token_id=model.generation_config.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id,
        )

        result = model.tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        return result.strip()
    except Exception as e:
        print(f"[!] Model inference failed: {e}")
        return "Failed"
