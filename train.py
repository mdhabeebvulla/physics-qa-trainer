import os
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch

def process_all_pdfs(data_dir="data"):
    qna_pairs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            qna_pairs += extract_qna_from_pdf(os.path.join(data_dir, filename))
    return qna_pairs

def train_pipeline():
    # Process all PDFs in data directory
    qa_pairs = process_all_pdfs()
    
    # Prepare dataset
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token
    dataset = prepare_dataset(qa_pairs, tokenizer)
    
    # Initialize model
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    
    # Training setup
    training_args = TrainingArguments(
        output_dir="./trained_model",
        num_train_epochs=15,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir="./logs",
        fp16=torch.cuda.is_available(),
        report_to="none",
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    # Start training
    trainer.train()
    model.save_pretrained("./trained_model")
    tokenizer.save_pretrained("./trained_model")

if __name__ == "__main__":
    train_pipeline()