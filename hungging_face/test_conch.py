import torch
from urllib.request import urlopen
from PIL import Image
from conch.open_clip_custom import create_model_from_pretrained
from conch.open_clip_custom import tokenize, get_tokenizer


model, preprocess = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token="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")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device).eval()

# 2) 定义类别
template = 'this is a photo of '
labels = [
    'adenocarcinoma histopathology',
    'brain MRI',
    'covid line chart',
    'squamous cell carcinoma histopathology',
    'immunohistochemistry histopathology',
    'bone X-ray',
    'chest X-ray',
    'pie chart',
    'hematoxylin and eosin histopathology'
]

# 3) 文本 → tokenized_prompts
tokenizer = get_tokenizer()  # 注意：这里要“调用”，不是赋值函数本身
prompts = [template + t for t in labels]

enc = tokenizer(
    prompts,
    padding="max_length",
    truncation=True,
    max_length=77,          # CLIP 常用长度；如你的模型有特别要求可改
    return_tensors="pt",
)
tokenized_prompts = enc["input_ids"].to(device)  # 传给 encode_text 的必须是 LongTensor[batch, seq_len]


# 4) 图片 → Tensor
dataset_url = 'https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/example_data/biomed_image_classification_example_data/'
test_imgs = [
    'squamous_cell_carcinoma_histopathology.jpeg',
    'H_and_E_histopathology.jpg',
    'bone_X-ray.jpg',
    'adenocarcinoma_histopathology.jpg',
    'covid_line_chart.png',
    'IHC_histopathology.jpg',
    'chest_X-ray.jpg',
    'brain_MRI.jpg',
    'pie_chart.png'
]

images = torch.stack([
    preprocess(Image.open(urlopen(dataset_url + img)).convert("RGB"))
    for img in test_imgs
]).to(device)

# 5) 前向推理
with torch.inference_mode():
    image_embeddings = model.encode_image(images)   # 不传 proj_contrast/normalize
    text_embeddings  = model.encode_text(tokenized_prompts)

    # 手动归一化（保证余弦相似度正确）
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings  = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

    logits = image_embeddings @ text_embeddings.T
    probs = logits.softmax(dim=-1)


# 6) 打印结果（按分数从大到小排序）
for idx, img_name in enumerate(test_imgs):
    print(f"\nResults for {img_name}:")
    sorted_indices = probs[idx].argsort(descending=True).tolist()
    for i in sorted_indices:
        print(f"  {labels[i]}: {probs[idx, i].item():.4f}")