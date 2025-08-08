import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config

model_id = "openai/gpt-oss-20b"
save_path = './gpt-oss-model-local'
try:
    print("최후의 방법을 시작합니다: 모델을 CPU에 강제 할당합니다.")
    print("이 과정은 매우 느리고 컴퓨터가 일시적으로 멈춘 것처럼 보일 수 있습니다.")

    # 양자화 설정
    quantization_config = Mxfp4Config(dequantize=True)

    # 오직 cpu에만 로드하여 분산 회피
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("모델 로드 완료.")

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("로컬에 저장 완료.")

except Exception as e:
    print(e)
