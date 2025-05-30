import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer


def init_model(cache_path):
    model_path = cache_path if (cache_path is not None and cache_path!="None") else "openbmb/MiniCPM-Llama3-V-2_5"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto'
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.tokenizer = tokenizer
    return model


import torch

def get_response_concat(model, question, image_path_list, max_new_tokens=1024, temperature=1.0):
    msgs = []
    system_prompt = 'Answer in detail.'
    if system_prompt:
        msgs.append(dict(type='text', value=system_prompt))
    if isinstance(image_path_list, list):
        msgs.extend([dict(type='image', value=p) for p in image_path_list])
    else:
        msgs = [dict(type='image', value=image_path_list)]
    msgs.append(dict(type='text', value=question))

    content = []
    for x in msgs:
        if x['type'] == 'text':
            content.append(x['value'])
        elif x['type'] == 'image':
            image = Image.open(x['value']).convert('RGB')
            content.append(image)

    # 包装为 chat 所需格式
    msgs = [{'role': 'user', 'content': content}]

    # 明确指定模型主设备
    device = next(model.parameters()).device

    with torch.cuda.amp.autocast():
        # 有些模型需要明确 device 指定，可在内部传递给 image encoder 时加 .to(device)
        # 如果你的模型内部未处理这一步，可 patch 一下 model.chat 内部逻辑或手动编码图片送入

        # 运行模型
        res = model.chat(
            msgs=msgs,
            context=None,
            image=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature != 0.0),
            tokenizer=model.tokenizer,
        )

    return res
