# ClinicalBERT-Project
Simulated clinical NLP project using ClinicalBERT to classify nursing notes for quantifiable workload and reportable quality and safety data. This project allows me to combine my experience as a Registered Nurse with my growing skills in data science and machine learning.


# ClinicalBERT Fall Risk Classifier

This project demonstrates the use of [ClinicalBERT]( https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT ), a domain-adapted BERT model, to classify clinical notes for patient fall risk. It is intended as a learning project for clinical natural language processing (NLP) and machine learning applications in healthcare.

---

## 🧠 Project Overview

- **Goal**: Automatically classify short clinical notes as either high fall risk (`label = 1`) or not (`label = 0`).
- **Data**: 100 manually labeled clinical notes representing realistic inpatient documentation scenarios.
- **Model**: Fine-tuned `emilyalsentzer/Bio_ClinicalBERT` for binary classification.
- **Use Case**: Simulated hospital safety monitoring tool to identify fall-prone patients from narrative chart entries.

---

## 🗃️ Dataset Structure

The dataset is in CSV format with the following columns:

| Column | Description                            |
|--------|----------------------------------------|
| note   | Row ID (not used in model)             |
| text   | Clinical note (free text)              |
| label  | 0 = No fall risk, 1 = High fall risk   |

---

## 🚀 Setup Instructions

1. **Clone the Repo**

```bash
git clone https://github.com/splanell/clinicalbert-fallrisk.git
cd clinicalbert-fallrisk
```

2. **Set up the Environment**

```bash
python3.10 -m venv clinicalbert-env
source clinicalbert-env/bin/activate
pip install -r requirements.txt
```

> ⚠️ Requires Python 3.10

3. **Install Dependencies **

```bash
pip install pandas datasets scikit-learn transformers evaluate
```

---

## ⚙️ Running the Classifier

### 1. Prepare Data

Place your `dataset_expanded.csv` in the project root.

### 2. Tokenize and Split

```python
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv("dataset_expanded.csv")
dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

tokenized = dataset.map(tokenize_function, batched=True)
split = tokenized.train_test_split(test_size=0.2)
```

### 3. Train the Model

```python
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
model = AutoModelForSequenceClassification.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT", num_labels=2
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
)
```

```python
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
```

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    compute_metrics=compute_metrics
)

trainer.train()
```

---

## 📈 Output

- Trained model saved to `./results`
- Evaluation metrics (accuracy, precision, recall, F1) reported during training

---

## 🧪 Example Use

```python
note = "Patient reports dizziness during ambulation. Needs assistance to transfer."
inputs = tokenizer(note, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax().item()
print("Fall Risk" if prediction == 1 else "No Fall Risk")
```

---

## 🏥 Clinical Disclaimer

> This project is for educational and prototyping purposes only. It is **not a medical device** and should not be used for clinical decision-making.

---

## 📚 References

- Alsentzer et al., *"Publicly Available Clinical BERT Embeddings"*, 2019.
- HuggingFace Transformers Library
- scikit-learn, pandas, datasets

---

## 🤝 Contributing

Feel free to fork, use, or modify this project for your own clinical NLP experiments. Pull requests welcome!

---
