from ctransformers import AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BertTokenizer, AutoTokenizer, pipeline, Text2TextGenerationPipeline
import torch
from peft import PeftModel

device = "cuda:0"

# Load Llama model
def load_llama_model(access_token):
    model = AutoModelForCausalLM.from_pretrained(
        "SinpxAI/Llama2-Chinese-7B-Chat-GGUF",
        model_file="llama2-chinese-7b-chat.Q4_K_M.gguf",
        model_type="llama",
        gpu_layers=0,
        hf=True
    )
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=access_token, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
    return pipe

# Load Bart model
def load_bart_model():
    base_model = "fnlp/bart-base-chinese"
    base_model_reload = BartForConditionalGeneration.from_pretrained(
        base_model,
        return_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto"
    ).half()
    model = PeftModel.from_pretrained(base_model_reload, "tonyma163/bart_v1")
    model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    #pipe = Text2TextGenerationPipeline(model=model, tokenizer=tokenizer)
    return model, tokenizer
