import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import gc

import pandas as pd

# 사용할 모델 경로
model_path = './gpt-oss-model-local'

print('모델 로드 시작')
# 4비트 양자화 설정 (BNB - BitsAndBytes)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,       # VRAM을 더 절약하기 위한 이중 양자화 옵션
    bnb_4bit_quant_type="nf4",            # NF4 (4-bit NormalFloat) 포맷 사용
    bnb_4bit_compute_dtype=torch.bfloat16, # 계산 시에는 bfloat16 데이터 타입을 사용해 정확도 손실 최소화
    llm_int8_enable_fp32_cpu_offload=True
)

# 총 24개 레이어 중 20개(0~19)는 GPU에, 나머지(20~23)는 CPU로
num_gpu_layers = 15
num_total_layers = 24
MAX_NEW_TOKEN = 1

device_map = {
    # 1. 임베딩은 무조건 GPU에
    "model.embed_tokens": 0,
    # 2. 앞부분 레이어들을 GPU(0번)에 할당
    **{f"model.layers.{i}": 0 for i in range(num_gpu_layers)},
    # 3. 뒷부분 레이어들을 CPU에 할당
    **{f"model.layers.{i}": "cpu" for i in range(num_gpu_layers, num_total_layers)},
    # 4. 최종 norm과 lm_head도 CPU에
    "model.norm": "cpu",
    "lm_head": "cpu"
}

# 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map=device_map
)

print('모델 로드 완료')

print('토크나이저 로드 시작')
tokenizer = AutoTokenizer.from_pretrained(model_path)

print('토크나이저 로드 완료')

# 대화 형식으로 질문 준비
messages = [
    {"role": "user", "content": "Explain what MXFP4 quantization is in simple terms."},
]

# 토크나이저로 모델이 이해할 수 있는 입력으로 변환
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device) # 모델이 올라가 있는 장치(GPU/CPU)로 입력을 보냄


gc.collect()
torch.cuda.empty_cache()

print('캐시 비우기 완료')

# # 텍스트 생성 실행
# # VRAM을 넘어선 부분은 RAM을 사용하므로 여기서 속도가 느려질 수 있습니다.
# outputs = model.generate(**inputs, max_new_tokens=256)
#
# # 결과 출력
# response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# # "assistant" 이후의 답변 부분만 깔끔하게 출력
# print("\n--- 모델 답변 ---")
# print(response_text.split("assistant\n")[-1])
#
messages = [
    {"role": "user", "content": "MXFP4 양자화에 대해 설명해줘."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

execution_times = []

for i in range(10) :

    print("추론 시작")
    start_time = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKEN,
        temperature=0.7
    )
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    execution_times.append(elapsed_time)
    print("추론 완료")


    print(tokenizer.decode(outputs[0]))
    print("\n" + "="*30)
    print(f"총 추론 시간: {elapsed_time:.2f}초")
    print("="*30)

df = pd.DataFrame(execution_times, columns=['elapsed_time'])
df.to_excel(f'./token_{MAX_NEW_TOKEN}_layer_{num_gpu_layers}.xlsx')


# messages = [
#     {"role": "system", "content": "Always respond in riddles"},
#     {"role": "user", "content": "What is the weather like in Madrid?"},
# ]
#
# inputs = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# ).to(model.device)
#
# generated = model.generate(**inputs, max_new_tokens=500)
# print(tokenizer.decode(generated[0][inputs["input_ids"].shape[-1]:]))