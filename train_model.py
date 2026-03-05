import pandas as pd
import numpy as np
import torch
import os
from datasets import Dataset, DatasetDict, ClassLabel, Features
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# --- 1. Load Dataset ---
csv_path = os.path.join("data", "legal_contract_clauses.csv")
if not os.path.exists(csv_path):
    print(f"Error: '{csv_path}' not found locally.")
    import sys
    sys.exit(1)

print("Step 1: Loading dataset...")
df = pd.read_csv(csv_path)
TEXT_COLUMN = "clause_text"
LABEL_COLUMN = "risk_level"

# --- 2. Create Label Mappings ---
print("Step 2: Creating label mappings...")
labels = df[LABEL_COLUMN].unique()
labels.sort() 
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
num_labels = len(labels)
df['label'] = df[LABEL_COLUMN].map(label2id)

# --- 3. Convert to Hugging Face Dataset ---
print("Step 3: Converting to Hugging Face Dataset...")
dataset = Dataset.from_pandas(df)

print("Step 3a: Casting 'label' column to ClassLabel for stratification...")
label_names_in_order = [id2label[i] for i in range(num_labels)]
class_label_feature = ClassLabel(names=label_names_in_order)
dataset = dataset.cast_column("label", class_label_feature)

print("Step 3b: Splitting dataset...")
train_test_split = dataset.train_test_split(test_size=0.2, seed=42, stratify_by_column="label")
dataset_dict = DatasetDict({
    'train': train_test_split['train'],
    'test': train_test_split['test']
})
print(f"Training data: {len(dataset_dict['train'])} examples")
print(f"Test data: {len(dataset_dict['test'])} examples")

# --- 4. Loading Model ---
print("\nStep 4: Loading model and tokenizer (nlpaueb/legal-bert-base-uncased)...")
model_name = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# --- 5. Tokenizing ---
print("\nStep 5: Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples[TEXT_COLUMN], padding="max_length", truncation=True)

tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
columns_to_remove = [TEXT_COLUMN, LABEL_COLUMN]
if "__index_level_0__" in tokenized_datasets["train"].column_names:
    columns_to_remove.append("__index_level_0__")
    
tokenized_datasets = tokenized_datasets.remove_columns(columns_to_remove)
tokenized_datasets.set_format("torch")
print("Tokenization complete.")

# --- 6. Metrics ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# --- 7. Set Training Arguments & Train ---
print("\nStep 6: Setting training arguments...")
model_output_dir = "./legal_bert_finetuned_risk"
training_args = TrainingArguments(
    output_dir=model_output_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
    report_to="none",
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("\nStep 7: Starting model training...")
trainer.train()
print("Training finished.")

# --- 8. Save ---
trainer.save_model(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
print(f"Model saved to {model_output_dir}")

# --- 9. Inference Test ---
print("\nStep 8: Quick inference test...")
risk_classifier = pipeline(
    "text-classification", 
    model=model_output_dir,
    device=0 if torch.cuda.is_available() else -1 
)

test_clause = "Indemnification. The Contractor agrees to indemnify..."
result = risk_classifier(test_clause, return_all_scores=True)
print(f"Test Clause: '{test_clause}'")
print(f"Prediction: {result}")
                                                                