import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("trained_model")
        model = AutoModelForCausalLM.from_pretrained("trained_model")
        return model, tokenizer
    except:
        return None, None

def main():
    st.title("Physics Q&A Assistant")
    
    model, tokenizer = load_model()
    
    if model is None:
        st.warning("Model not trained yet! Add PDFs to the data folder and push to GitHub.")
        return
    
    # Question input
    question = st.text_input("Ask a physics question:")
    
    if question:
        inputs = tokenizer.encode(question, return_tensors="pt")
        outputs = model.generate(
            inputs,
            max_length=150,
            temperature=0.7,
            do_sample=True
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Answer:")
        st.write(answer.split("A:")[-1].strip())

if __name__ == "__main__":
    main()