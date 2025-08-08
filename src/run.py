import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GptOssForCausalLM
from transformers.utils.quantization_config import Mxfp4Config


# 사용할 모델 이름
model_name = "openai/gpt-oss-20b"

# 4비트 양자화 설정 (BNB - BitsAndBytes)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,       # VRAM을 더 절약하기 위한 이중 양자화 옵션
    bnb_4bit_quant_type="nf4",            # NF4 (4-bit NormalFloat) 포맷 사용
    bnb_4bit_compute_dtype=torch.bfloat16 # 계산 시에는 bfloat16 데이터 타입을 사용해 정확도 손실 최소화
)

print("모델과 토크나이저를 로딩합니다... (시간이 걸릴 수 있습니다)")

quantization_config = Mxfp4Config(dequantize=True)

model = GptOssForCausalLM.from_pretrained(model_name,
                                          quantization_config=quantization_config,
                                          torch_dtype="auto",
                                          device_map="auto"
                                          )

# 설정과 함께 모델 로드
# device_map="auto"는 VRAM과 RAM에 모델을 자동으로 나누어 로드하는 핵심 옵션입니다.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

print("모델 로딩 완료! 텍스트 생성을 시작합니다.")

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

# 텍스트 생성 실행
# VRAM을 넘어선 부분은 RAM을 사용하므로 여기서 속도가 느려질 수 있습니다.
outputs = model.generate(**inputs, max_new_tokens=256)

# 결과 출력
response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# "assistant" 이후의 답변 부분만 깔끔하게 출력
print("\n--- 모델 답변 ---")
print(response_text.split("assistant\n")[-1])