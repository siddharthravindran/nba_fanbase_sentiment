"""Score ingested docs with the fine-tuned RoBERTa sentiment model."""
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Siddharthr30/emotion-model"


def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


def score(text: str, tokenizer, model, threshold: float = 0.4):
    inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].numpy()
    id2label = model.config.id2label
    scored = sorted(
        [(id2label[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1],
        reverse=True,
    )
    predicted = [label for label, p in scored if p >= threshold]
    return {"labels": predicted, "scores": dict(scored)}


if __name__ == "__main__":
    tokenizer, model = load_model()
    print(score("This trade is an absolute disaster for our front office.", tokenizer, model))
