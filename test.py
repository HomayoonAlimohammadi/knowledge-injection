import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from knowledge_dict import knowledge_dict
from train import inject_knowledge

def test(model, tokenizer):
    test_df = pd.read_csv("data/test.csv")
    test_dataset = Dataset.from_pandas(test_df)

    model.eval()

    for sample in test_dataset:
        text = sample["text"]
        ground_truth = sample["label"]

        # Inject knowledge
        injected_text = inject_knowledge(text, knowledge_dict)

        inputs = tokenizer(
            injected_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        print(f"Text: {text}")
        print(f"Injected: {injected_text}")
        print(f"Predicted Class: {predicted_class}, Ground Truth: {ground_truth}")
        print("-"*70)
