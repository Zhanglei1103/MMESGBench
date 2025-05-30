# models/deepseek_vl_chat.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
import torch

def init_model(cache_path=None):
    model_path = cache_path if (cache_path and cache_path != "None") else "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = model.to(torch.bfloat16).to("cuda:1").eval()
    model.tokenizer = tokenizer
    model.vl_chat_processor = vl_chat_processor
    return model

def get_response_concat(model, question, image_path_list, max_new_tokens=512, temperature=0.7):
    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder> " * len(image_path_list) + question,
            "images": image_path_list
        },
        {
            "role": "Assistant",
            "content": ""
        }
    ]
    pil_images = load_pil_images(conversation)
    inputs = model.vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(model.device)

    with torch.no_grad():
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=model.tokenizer.eos_token_id,
            bos_token_id=model.tokenizer.bos_token_id,
            eos_token_id=model.tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature
        )

    answer = model.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer
