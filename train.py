import os
import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
import pdfplumber
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
import torch
# Add this function BEFORE process_all_pdfs()
def extract_qna_from_pdf(pdf_path):
    qna_pairs = []
    current_question = ""
    current_answer = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith("Q:"):
                    if current_question and current_answer:
                        qna_pairs.append({
                            "prompt": current_question,
                            "completion": current_answer
                        })
                    current_question = line[2:].strip()
                    current_answer = ""
                elif line.startswith("A:"):
                    current_answer = line[2:].strip()
                elif current_answer != "":
                    current_answer += " " + line.strip()
            # Add final Q&A pair from each page
            if current_question and current_answer:
                qna_pairs.append({
                    "prompt": current_question,
                    "completion": current_answer
                })
    return qna_pairs
def process_all_pdfs(data_dir="data"):
    qna_pairs = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            qna_pairs += extract_qna_from_pdf(os.path.join(data_dir, filename))
    return qna_pairs
def prepare_dataset(qa_pairs, tokenizer):  # Accept tokenizer as argument
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in qa_pairs:
        full_input = f"{item['prompt']} {item['completion']}{tokenizer.eos_token}"
        encodings = tokenizer(
            full_input,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        input_ids_list.append(encodings["input_ids"])
        attention_mask_list.append(encodings["attention_mask"])
        labels_list.append(encodings["input_ids"])

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    })

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
        num_train_epochs=3,  # Reduced from 15
        per_device_train_batch_size=2,  # Reduced from 4
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,  # Save memory
        optim="adafactor",  # Lighter optimizer
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",  # Disable saving during training
        report_to="none",
        fp16=False,  # Disable for CPU
        dataloader_pin_memory=False,
        dataloader_num_workers=2,
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
