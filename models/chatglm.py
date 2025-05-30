from transformers import AutoTokenizer, AutoModel

def init_model(cache_path=None):
    model_path = cache_path if (cache_path and cache_path != "None") else "THUDM/chatglm3-6b-128k"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda().eval()
    model.tokenizer = tokenizer
    return model


def get_response_concat(model, question, context, max_new_tokens=1024, temperature=0.7):
    prompt = f"""Please answer the question based on the context below.

Context:
{context}

Question: {question}
Answer:"""

    try:
        # ChatGLM 的 chat 接口自动处理 history 和 prompt 格式
        response, _ = model.chat(
            model.tokenizer,
            prompt,
            history=[],
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
        )
    except Exception as e:
        print(f"[!] Model inference failed: {e}")
        response = "Failed"
    return response
