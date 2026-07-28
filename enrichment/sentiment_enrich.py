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


def score(text: str, tokenizer, model):
    inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].numpy()
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
