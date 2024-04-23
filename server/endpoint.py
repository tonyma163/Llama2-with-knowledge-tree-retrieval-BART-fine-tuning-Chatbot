from fastapi import APIRouter
from pydantic import BaseModel
import re

#
from models import load_llama_model, load_bart_model, query_llama
from knowledgeTree import load_knowledge_set, build_tree
from utils import retrieve_answers

router = APIRouter()

# Hugging Face Access Token
hf_access_token = "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"

# Load Models
llama_pipe = load_llama_model(hf_access_token) # Llama pipeline
bart_model, bart_tokenizer = load_bart_model() # Bart model & tokenizer

# Knowledge set file path
file_path = "../dataset/knowledge_set.txt"

# Load knowledge set
knowledge_df = load_knowledge_set(file_path)

# Build knowledge tree
G = build_tree(knowledge_df)

# Llama2 Prompting
system_prompt = """
Please answer the following question based on the provided context.
Provide only the direct answers without any additional explanations or context.
Please only output answers only.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer, please don't share false information.
"""

class UserInput(BaseModel):
    text: str

@router.post("/generate/")
def generate_text(user_input: UserInput):
    # Retrieve the latest user input
    match = re.search(r'##@(.*?)@##', user_input.text)
    if match:
        # Extracts text found between the markers
        query = match.group(1)
        full_input_without_markers = re.sub(r'##@.*?@##', query, user_input.text)
    else:
        query = user_input.text
        full_input_without_markers = user_input.text
        
    # Attempt to retrieve from the knowledge set for Llama2
    answers, require_llama = retrieve_answers(G, query)
    
    # If answers are found in the graph, and further processing with Llama2 is required
    if answers and require_llama:
        return {"response": query_llama(llama_pipe, system_prompt, query, answers)}
    elif answers and not require_llama:
        return {"response": answers}
    else:
        # If no answers are found, use BART model to generate an answer from the full context without markers
        inputs = bart_tokenizer.encode(full_input_without_markers, return_tensors="pt")
        outputs = bart_model.generate(input_ids=inputs.to("cuda:0"), max_new_tokens=126)
        response_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text.replace(' ', '')
        return {"response": response_text}