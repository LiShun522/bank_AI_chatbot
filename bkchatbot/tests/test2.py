# import torch

# # 檢查 CUDA 是否可用
# print(torch.cuda.is_available())  # 應該回傳 True

# # 檢查可用的 GPU 數量
# print(torch.cuda.device_count())  # 應該至少回傳 1

# # 確認 GPU 名稱
# print(torch.cuda.get_device_name(0))  # 應該顯示您的 GPU 型號，例如 "NVIDIA GeForce RTX 4060"


# import torch
# x = torch.rand(9, 9).cuda()
# print(x)



# import torch
# import time

# # 在 CPU 運行
# x_cpu = torch.rand(10000, 10000)
# start = time.time()
# y_cpu = torch.matmul(x_cpu, x_cpu)
# print(f"CPU 計算時間: {time.time() - start:.4f} 秒")

# # 在 GPU 運行
# x_gpu = torch.rand(10000, 10000, device="cuda")
# start = time.time()
# y_gpu = torch.matmul(x_gpu, x_gpu)
# print(f"GPU 計算時間: {time.time() - start:.4f} 秒")

import torch
print(torch.cuda.is_available())  # 應該返回 True，表示 CUDA 可用




from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"

# 啟用 4-bit 量化設定
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

# 下載 Tokenizer & 量化模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# 測試對話
inputs = tokenizer("請用中文介紹自己。", return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))



