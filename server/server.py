from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import ast
import networkx as nx
import jieba.posseg as pseg
import jieba
import random
import re
from fastapi import Depends

#
from Models import load_llama_model, load_bart_model
from KnowledgeTree import load_knowledge_set, build_tree

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

# Hugging Face Access Token
hf_access_token = "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"

# Load Models
llama_pipe = load_llama_model(hf_access_token) # Llama pipeline
bart_model, bart_tokenizer = load_bart_model() # Bart model & tokenizer

# Knowledge set file path
file_path = "../dataset/knowledge_set.txt"

# Load knowledge set
knowledge_df = load_knowledge_set(file_path)
G = build_tree(knowledge_df)

# Part of Speech Function
def checkSentence(sentence):
    separated_sentences = []
    temp_sentence = ""
    in_bracket = False

    for char in sentence:
        if char == "『" or char == "《":
            in_bracket = True
            if temp_sentence:
                separated_sentences.append(temp_sentence)
                temp_sentence = ""
            temp_sentence += char
        elif char == "』" or char == "》":
            in_bracket = False
            temp_sentence += char
            separated_sentences.append(temp_sentence)
            temp_sentence = ""
        elif char == "。" and not in_bracket:
            separated_sentences.append(temp_sentence)
            temp_sentence = ""
        else:
            temp_sentence += char

    if temp_sentence:
        separated_sentences.append(temp_sentence)

    return separated_sentences

def extractPOS(sentences):

    result = []
    special_words = []

    for sentence in sentences:
        if '『' in sentence or '《' in sentence:
            # Remove the brackets
            sentence = sentence.replace('『', '@').replace('』', '@').replace('《', '@').replace('》', '@')
            #result.append(sentence)
            special_words.append(sentence)
        else:
            allowSentencePOS = ("a", "ad", "ag", "an", ## 形容词 
                                "b", # 区别词
                                "c", # 连词
                                "f", 
                                "g", 
                                "h",
                                "i", 
                                "j", 
                                "k", 
                                "l", 
                                "m", "mg", "mq", 
                                "n", "ng", "nr", "nrfg", "nrt","ns", "nt", "nz", 
                                "o", 
                                "q", 
                                "s", 
                                "t", "tg",
                                "v", "vd", "vg", "vi", "vn","vq" ## 动词
                                )

            exclude = ( "p", ## 介词
                "d", "df", "dg", # 副词
                "e", # 叹词 
                "r", "rg", "rr", "rz",  ## 代词
                "u",  "ud", "ug", "uj", "ul", "uv", "uz",  ## 助词
                "x",  ## 非语素词（包含标点符号
                "y", ## 语气词
                "z","zg" ## 助词
              )

            sentence = pseg.lcut(sentence)
            seg_words = [word for word, pos in sentence if pos in allowSentencePOS]
            result.extend(seg_words)
            
    return special_words+result

# Retrieve Data from Knowledge Tree
def has_common_character(keyword, category):
    return any(char in category for char in keyword)

def retrieve_answers(graph, query):
    segmented_keywords = extractPOS(checkSentence(query))
    special_keywords = [kw.strip('@') for kw in segmented_keywords if kw.startswith('@')]
    regular_keywords = [kw for kw in segmented_keywords if not kw.startswith('@')]

    category_answers = {}  # Store answers grouped by categories
    special_categories = {'新闻', '评论'}

    # Top-Down Search: From entities to categories to answers
    for entity_node in (n for n in graph.nodes if graph.nodes[n]['type'] == 'Entity'):
        entity_keywords = [kw for kw in regular_keywords if kw == graph.nodes[entity_node]['name']]
        if entity_keywords:
            for category_node in graph.successors(entity_node):
                category_keywords = [kw for kw in regular_keywords if has_common_character(kw, graph.nodes[category_node]['name'])]
                if category_keywords:
                    for answer_node in graph.successors(category_node):
                        category_name = graph.nodes[category_node]['name']
                        answer_content = graph.nodes[answer_node]['content']
                        if category_name not in category_answers:
                            category_answers[category_name] = []
                        if category_name not in special_categories:
                            category_answers[category_name].append(category_name+answer_content)
                        else:
                            category_answers[category_name].append(answer_content)

    # Bottom-Up Search: From answers to categories to entities
    for answer_node in (n for n in graph.nodes if graph.nodes[n].get('type') == 'Answer'):
        if any(kw == graph.nodes[answer_node]['content'] for kw in special_keywords):
            for category_node in graph.predecessors(answer_node):
                category_name = graph.nodes[category_node]['name']
                if any(has_common_character(kw, category_name) for kw in regular_keywords):
                    for entity_node in graph.predecessors(category_node):
                        if category_name not in category_answers:
                            category_answers[category_name] = []
                        category_answers[category_name].append(graph.nodes[entity_node]['name'] + category_name)

    final_answers = []
    require_llama = False
    for category, answers in category_answers.items():
        if category in special_categories:
            # Randomly pick one answer if category is special
            final_answers.append(random.choice(answers))
        else:
            # Collect all answers for other categories
            final_answers.extend(answers)
            require_llama = True  # Indicate further processing may be needed for non-special categories

    return final_answers, require_llama

# Llama2 (Knowledge Tree) + Fine-tuned Bart
system_prompt = """
Please answer the following question based on the provided context.
Provide only the direct answers without any additional explanations or context.
Please only output answers only.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer, please don't share false information.
"""

def query_llama(query, context):
    # Test
    prompt = f"""
    <s>[INST] <<SYS>>
    {system_prompt}
     <</SYS>> [/INST]</s>

    <s>[INST]
    Question: {query}
    Context: {context}
    Answer:
    [/INST]
        """

    output = llama_pipe(
        prompt,
        do_sample=True,
        max_new_tokens=256,
        top_k=40,
        top_p=0.95,
        temperature=0.75,
        return_full_text=False,
        repetition_penalty=1
    )
    return output[0]['generated_text']

class UserInput(BaseModel):
    text: str

@app.post("/generate/")
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
        return {"response": query_llama(query, answers)}
    elif answers and not require_llama:
        return {"response": answers}
    else:
        # If no answers are found, use BART model to generate an answer from the full context without markers
        inputs = bart_tokenizer.encode(full_input_without_markers, return_tensors="pt")
        outputs = bart_model.generate(input_ids=inputs.to("auto"), max_new_tokens=126)
        response_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = response_text.replace(' ', '')
        return {"response": response_text}