from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델과 토크나이저 로드
model_name = "gpt2"  # 원하는 모델 이름으로 변경 가능
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 패딩 방향 설정
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_name)

# 입력 텍스트 배치
input_texts = ["Hello, how are you?", "What is your name?", "Where do you live?"]
inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

# 텍스트 생성
output_sequences = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=50,  # 생성할 텍스트의 최대 길이
    num_return_sequences=1,  # 각 입력에 대해 생성할 텍스트의 수
)

# 결과 디코딩
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in output_sequences]
for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {text}")
