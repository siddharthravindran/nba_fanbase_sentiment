"""Score ingested docs with the fine-tuned RoBERTa sentiment model (v2:
sports-specific taxonomy, single-label softmax classifier)."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Siddharthr30/emotion-model-v2"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


def pick_device():
    """Apple Silicon GPU when present, else CPU. Batched MPS inference runs
    ~5x faster than the one-at-a-time CPU path with identical labels."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def score_batch(texts: list[str], tokenizer, model, device=None) -> list[dict]:
    """Same output shape as score(), one dict per input text. Padding to the
    longest item in the batch is what makes this worth doing - the per-call
    overhead of score() dominates its runtime at this document length."""
    device = device or next(model.parameters()).device
    inputs = tokenizer(
        texts, truncation=True, max_length=64, padding=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()

    id2label = model.config.id2label
    results = []
    for row in probs:
        scored = sorted(
            [(id2label[i], float(row[i])) for i in range(len(row))],
            key=lambda x: x[1],
            reverse=True,
        )
        top_label, top_score = scored[0]
        results.append(
            {"top_emotion": top_label, "top_score": top_score, "scores": dict(scored)}
        )
    return results


def score(text: str, tokenizer, model):
    device = next(model.parameters()).device
    inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].cpu().numpy()
    id2label = model.config.id2label
    scored = sorted(
        [(id2label[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True,
    )
    top_label, top_score = scored[0]
    return {"top_emotion": top_label, "top_score": top_score, "scores": dict(scored)}


if __name__ == "__main__":
    tokenizer, model = load_model()
    print(score("This trade is an absolute disaster for our front office.", tokenizer, model))
