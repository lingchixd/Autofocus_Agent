from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from PIL import Image
import requests
import torch

# 1. 加载 processor 和 model
processor = AutoProcessor.from_pretrained("vinid/plip")
model = AutoModelForZeroShotImageClassification.from_pretrained("vinid/plip")

# 2. 加载图片
image = Image.open(r"C:\Users\Lingchi Deng\Pictures\Guilin_Stone.jpg")

# 3. 定义候选标签
candidate_labels = ["animals", "humans", "hills"]

# 4. 预处理（把图像 + 文本转成模型输入）
inputs = processor(images=image, text=candidate_labels, return_tensors="pt", padding=True)

# 5. 前向传播
with torch.no_grad():
    outputs = model(**inputs)

# 6. 提取 logit 并算 softmax 概率
probs = outputs.logits_per_image.softmax(dim=1)

# 7. 打印结果
for label, prob in zip(candidate_labels, probs[0].tolist()):
    print(f"{label}: {prob:.4f}")
