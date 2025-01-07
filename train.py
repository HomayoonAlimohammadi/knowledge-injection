import numpy as np
import pandas as pd
from transformers import (
    GPT2Tokenizer, 
    GPT2ForSequenceClassification, 
    TrainingArguments, 
    Trainer,
)
from datasets import Dataset, DatasetDict
import evaluate
from knowledge_dict import knowledge_dict

def inject_knowledge(text, knowledge_dict, delimiter="[KNOWLEDGE]"):
    """
    Inject domain-specific definitions or synonyms directly into the text.
    For each keyword in the knowledge dictionary found in 'text',
    append an inline note with the definition at the end of the text.
    """
    injected_text = text
    for keyword, definition in knowledge_dict.items():
        if keyword.lower() in text.lower():
            # Add knowledge injection
            injection_snippet = f" {delimiter} {keyword}: {definition}"
            injected_text += injection_snippet
    return injected_text

def preprocess_function_wrapper(tokenizer, knowledge_dict):
    def preprocess_function(examples):
        texts = examples["text"]
        # Inject domain knowledge into each text
        injected_texts = [inject_knowledge(t, knowledge_dict) for t in texts]
        
        # Tokenize
        tokenized = tokenizer(
            injected_texts, 
            padding="max_length", 
            max_length=64, 
            truncation=True
        )
        
        # The label is included as-is
        tokenized["labels"] = examples["label"]
        return tokenized

    return preprocess_function

def train():
    # Read CSV
    train_df = pd.read_csv("data/train.csv")
    valid_df = pd.read_csv("data/validation.csv")

    # Convert to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset
    })

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # GPT-2 default pad token is None, so we need to set a pad token
    # for sequence classification tasks
    tokenizer.pad_token = tokenizer.eos_token

    processed_dataset = dataset_dict.map(preprocess_function_wrapper(tokenizer, knowledge_dict), batched=True)

    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)

    # IMPORTANT: Because we set pad_token to eos_token for GPT-2
    # we need to resize the token embeddings
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="./_results",
        learning_rate=2e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=10,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",  # Avoid storing logs in default environment
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = metric.compute(predictions=predictions, references=labels)
        if acc is None:
            raise ValueError("Accuracy is None, check the metric computation.")
        return {"accuracy": acc["accuracy"]}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer.evaluate(), model, tokenizer
