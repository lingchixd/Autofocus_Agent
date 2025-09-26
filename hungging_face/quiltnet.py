import io, ssl
import torch
from urllib.request import urlopen, Request
from PIL import Image
import open_clip

# ========== 1) 模型与设备 ==========
device = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))

# QuiltNet-B-32（OpenCLIP 风格）——从 HF Hub 加载
model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
    "hf-hub:wisdomik/QuiltNet-B-32"
)
tokenizer = open_clip.get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")
model = model.to(device).eval()

# QuiltNet 的文本上下文长度为 77（CLIP 典型长度）
context_length = 77

# ========== 2) 类别 & 提示（复用 BiomedCLIP 示例）==========
template = "this is a photo of "
labels = [
    "adenocarcinoma histopathology",
    "brain MRI",
    "covid line chart",
    "squamous cell carcinoma histopathology",
    "immunohistochemistry histopathology",
    "bone X-ray",
    "chest X-ray",
    "pie chart",
    "hematoxylin and eosin histopathology",
]

# ========== 3) 测试图像（复用 BiomedCLIP 示例）==========
dataset_url = (
    "https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/"
    "resolve/main/example_data/biomed_image_classification_example_data/"
)
test_imgs = [
    "squamous_cell_carcinoma_histopathology.jpeg",
    "H_and_E_histopathology.jpg",
    "bone_X-ray.jpg",
    "adenocarcinoma_histopathology.jpg",
    "covid_line_chart.png",
    "IHC_histopathology.jpg",
    "chest_X-ray.jpg",
    "brain_MRI.jpg",
    "pie_chart.png",
]

# 可选：某些环境需要忽略证书验证
ssl._create_default_https_context = ssl._create_unverified_context

def fetch_image(url: str) -> Image.Image:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        img = Image.open(io.BytesIO(r.read())).convert("RGB")
    return img

# ========== 4) 预处理与打包 ==========
# 推理建议用 preprocess_val（确定性）
images = []
for name in test_imgs:
    img = fetch_image(dataset_url + name)
    images.append(preprocess_val(img))
images = torch.stack(images).to(device)

texts = tokenizer([template + l for l in labels], context_length=context_length).to(device)

# ========== 5) 前向推理 ==========
with torch.no_grad():
    image_features = model.encode_image(images)
    text_features  = model.encode_text(texts)

    # 归一化（确保在同一球面上算相似度）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)

    logits = image_features @ text_features.T
    probs = logits.softmax(dim=-1)

# ========== 6) 打印结果 ==========
probs = probs.detach().cpu()
sorted_indices = torch.argsort(probs, dim=-1, descending=True)
top_k = len(labels)  # 想只看 Top-1 就设为 1

for i, img_name in enumerate(test_imgs):
    print(f"\n{img_name}:")
    for j in range(top_k):
        idx = int(sorted_indices[i, j])
        print(f"  {labels[idx]}: {float(probs[i, idx]):.4f}")
