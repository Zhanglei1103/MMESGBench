# from openai import OpenAI
# client = OpenAI()


# def extract_answer(question, output, prompt, model_name="gpt-4o"):
#     try:
#         response = client.chat.completions.create(
#             model=model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": prompt,
#                 },
#                 {
#                 "role": "assistant",
#                 "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
#                 }
#             ],
#             temperature=0.0,
#             max_tokens=256,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         response = response.choices[0].message.content
#     except:
#         response = "Failed"
    
#     return response

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # Replace with actual API key if needed
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


def extract_answer(question, output, prompt, model_name="qwen-max"):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
            {
            "role": "assistant",
            "content": "\n\nQuestion:{}\nAnalysis:{}\n".format(question, output)
            }
        ],
        temperature=0.0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        seed=42
    )
    response = response.choices[0].message.content

    
    return response