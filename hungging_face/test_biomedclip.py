import io
import os
import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

# 1) 模型与设备
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(f"[Info] Using device: {device}")

# 下载/加载模型与预处理（首次会从 Hugging Face 拉到本地缓存）
model, preprocess = create_model_from_pretrained(MODEL_ID)
tokenizer = get_tokenizer(MODEL_ID)
model.to(device).eval()

# 2) 工具函数：加载本地或URL图片
def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        with urlopen(path_or_url) as r:
            img = Image.open(io.BytesIO(r.read()))
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Image not found: {path_or_url}")
        img = Image.open(path_or_url)
    return img.convert("RGB")

# 3) 两档版：清晰 vs 严重离焦 → 连续分数
@torch.no_grad()
def defocus_score_zeroshot(path_or_url: str, context_length: int = 256) -> dict:
    """
    返回 0~1 离焦分数：0=清晰，1=严重离焦。
    """
    prompts = [
        "a sharply focused biomedical image",
        "a severely out-of-focus (defocused) biomedical image",
    ]
    img_tensor = preprocess(load_image(path_or_url)).unsqueeze(0).to(device)
    text_tokens = tokenizer(prompts, context_length=context_length).to(device)

    image_features, text_features, logit_scale = model(img_tensor, text_tokens)
    probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1).squeeze(0)  # [2]

    p_sharp = float(probs[0].item())
    score = 1.0 - p_sharp  # 越大越离焦

    return {
        "sharpness_prob": p_sharp,           # 清晰的相对概率
        "defocus_prob": 1.0 - p_sharp,       # 离焦的相对概率
        "defocus_score_0to1": score          # 建议使用的连续分数
    }

# 4) 四档版：sharp/slight/moderate/severe → 连续分数（期望值）
@torch.no_grad()
def defocus_score_4level(path_or_url: str, context_length: int = 256) -> dict:
    prompts = [
        "a sharply focused biomedical image",            # 0.00
        "a slightly out-of-focus biomedical image",      # 0.33
        "a moderately out-of-focus biomedical image",    # 0.66
        "a severely out-of-focus biomedical image",      # 1.00
    ]
    img_tensor = preprocess(load_image(path_or_url)).unsqueeze(0).to(device)
    text_tokens = tokenizer(prompts, context_length=context_length).to(device)

    image_features, text_features, logit_scale = model(img_tensor, text_tokens)
    probs = (logit_scale * image_features @ text_features.t()).softmax(dim=-1).squeeze(0)  # [4]

    levels = torch.tensor([0.0, 1/3, 2/3, 1.0], device=probs.device)
    score = float((probs * levels).sum().item())
    top_idx = int(torch.argmax(probs).item())

    return {
        "defocus_score_0to1": score,
        "pred_level": ["sharp","slight","moderate","severe"][top_idx],
        "probs": [float(p) for p in probs.tolist()]
    }

# 5) 直接跑你这张图
def main():
    img_path = r"E:\SS316_ShotPeen\images\4\05_01_04.png"  # ← 你的图片
    print(f"[Info] Evaluating: {img_path}")

    r2 = defocus_score_zeroshot(img_path)
    print("\n[Two-level (sharp vs severe blur)]")
    print(f"  sharpness_prob     : {r2['sharpness_prob']:.4f}")
    print(f"  defocus_prob       : {r2['defocus_prob']:.4f}")
    print(f"  defocus_score_0to1 : {r2['defocus_score_0to1']:.4f}  (0=清晰, 1=严重离焦)")

    r4 = defocus_score_4level(img_path)
    print("\n[Four-level (sharp/slight/moderate/severe)]")
    print(f"  pred_level         : {r4['pred_level']}")
    print(f"  defocus_score_0to1 : {r4['defocus_score_0to1']:.4f}")
    print(f"  probs              : {r4['probs']}")

if __name__ == "__main__":
    main()
