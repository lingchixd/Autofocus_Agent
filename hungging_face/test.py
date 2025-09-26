# pip install -U datasets pillow open_clip_torch torch
import re, torch
from datasets import load_dataset
from PIL import Image
from collections import Counter
from open_clip import create_model_from_pretrained, get_tokenizer

# 1) 加载 uBench 测试集
ds = load_dataset("jnirschl/uBench", split="test")

# 2) 加载 BiomedCLIP（来自 HF Hub）
model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
model, preprocess = create_model_from_pretrained(model_id)
tokenizer = get_tokenizer(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
context_length = 256

# 3) 工具函数
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[\s]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)  # 去标点
    return s.strip()

def extract_qas(example):
    """把 example['questions'] 统一成 [{'question': str, 'answers': [str, ...]}, ...]"""
    q = example.get("questions")
    if q is None:
        return []
    out = []
    if isinstance(q, dict):
        # 优先常见键
        for k in ["vqa", "qa", "questions", "items"]:
            if k in q and isinstance(q[k], list):
                out = q[k]
                break
        if not out:  # 兜底：收集字典中的 list
            for v in q.values():
                if isinstance(v, list):
                    out.extend(v)
    elif isinstance(q, list):
        out = q
    # 统一字段名
    norm = []
    for it in out:
        if not isinstance(it, dict):
            continue
        ques = it.get("question") or it.get("query") or ""
        ans  = it.get("answers")  or it.get("answer") or []
        if isinstance(ans, str):
            ans = [ans]
        if ques and ans:
            norm.append({"question": ques, "answers": ans})
    return norm

# 4) 构建候选答案集合（从 test 的标注里抽频次最高的若干个，控制计算量）
answer_counter = Counter()
for ex in ds:
    for qa in extract_qas(ex):
        for a in qa["answers"]:
            answer_counter[normalize_text(a)] += 1

# 选择 top-K 作为闭集候选（很多生医 VQA 以 yes/no、解剖部位/模态词为主）
K = 128  # 可调；更大更准也更慢
answer_vocab = [a for a, _ in answer_counter.most_common(K)]

# 若某题的金标不在 top-K，预测永远错；可以适当增大 K 或按题目动态扩展备选
print(f"候选答案集大小: {len(answer_vocab)}")

# 5) 评测循环
total, correct = 0, 0

def predict_answer(image_pil: Image.Image, question: str, candidates: list[str]) -> str:
    # 文本假设：把问题与候选答案拼接，交给 CLIP 文本编码器
    prompts = [f"question: {question} answer: {a}" for a in candidates]
    # 编码图像
    image = preprocess(image_pil).unsqueeze(0).to(device)
    # 编码文本
    text = tokenizer(prompts, context_length=context_length).to(device)

    with torch.no_grad():
        image_features, text_features, logit_scale = model(image, text)
        # 相似度（1 x C）
        logits = (logit_scale * image_features @ text_features.t()).softmax(dim=-1)
        pred_idx = torch.argmax(logits, dim=-1).item()
    return candidates[pred_idx]

# 可先设 N=500 快速跑通，再全量评测
N = None  # e.g., 500

for i, ex in enumerate(ds):
    if N is not None and i >= N:
        break
    image = ex["image"]  # PIL.Image
    qas = extract_qas(ex)
    for qa in qas:
        q = qa["question"]
        gts = [normalize_text(a) for a in qa["answers"] if isinstance(a, str)]
        if not gts:
            continue

        # 预测
        pred = predict_answer(image, q, answer_vocab)
        pred_norm = normalize_text(pred)

        total += 1
        if pred_norm in set(gts):
            correct += 1

acc = correct / total if total else 0.0
print(f"BiomedCLIP 零样本 VQA accuracy: {acc:.4f}  ({correct}/{total})")
