from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
load_dotenv()

token = os.getenv("HF_TOKEN")


model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token) 
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
nlp_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
input_text = "The future of AI in healthcare is"
results = nlp_pipeline(input_text, max_new_tokens=50, temperature=0.7,num_return_sequences=2)
print(results)
# print(results[0]["generated_text"])
