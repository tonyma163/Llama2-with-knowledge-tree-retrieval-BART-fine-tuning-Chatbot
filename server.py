from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BartForConditionalGeneration, BertTokenizer
from peft import PeftModel
import torch

app = FastAPI()

origins = [
    "http://localhost:3000",
]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load your fine-tuned model
base_model = "fnlp/bart-base-chinese"
new_model = "tonyma163/bart_v1"

base_model_reload = BartForConditionalGeneration.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",
        #trust_remote_code=True,
)
base_model_reload.half()

model = PeftModel.from_pretrained(base_model_reload, new_model)
device="cuda:0"
model = model.to(device)
model.eval()  # Set to eval mode

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(base_model, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

class UserInput(BaseModel):
    text: str

@app.post("/generate/")
def generate_text(user_input: UserInput):
    inputs = tokenizer.encode(user_input.text, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(device), max_new_tokens=126)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response_text}