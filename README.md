# BERT Sentiment Classification

This project fine tunes a pretrained BERT model (bert-base-uncased) for multi class sentiment classification on short text data using PyTorch and Hugging Face Transformers.

## Dataset
The dataset consists of text samples labeled into multiple sentiment categories such as happy, sad, angry, surprise, disgust, and not relevant. Noisy labels and irrelevant categories were removed, and a stratified train validation split was used to preserve class distribution.  

The processed dataset is included in this repository as a CSV file under the `content/` directory for reproducibility.

## Methodology
Text data is tokenized using the BERT tokenizer with attention masks. A pretrained BERT sequence classification model is fine tuned on the training set. Model performance is evaluated on a validation set using weighted F1 score to account for class imbalance.

## Model & Training
- Model: bert-base-uncased  
- Framework: PyTorch, Hugging Face Transformers  
- Max sequence length: 256  
- Optimizer: AdamW (lr = 1e-5)  
- Scheduler: Linear learning rate decay  
- Epochs: 10  
- Batch size: 4 (train), 32 (validation)

## Results
The model achieves a weighted F1 score of approximately 0.83–0.84 on the validation set, indicating effective learning with no data leakage.

## Setup
```bash
pip install -r requirements.txt