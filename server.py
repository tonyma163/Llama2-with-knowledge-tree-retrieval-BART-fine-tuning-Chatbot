from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ctransformers import AutoModelForCausalLM
from transformers import BartForConditionalGeneration, BertTokenizer, AutoTokenizer, pipeline, Text2TextGenerationPipeline
from peft import PeftModel
import torch
import pandas as pd
import ast
import networkx as nx
import jieba.posseg as pseg
import jieba
import random
import re

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

# Load Llama2 for knowledge retrieval
model_llama = AutoModelForCausalLM.from_pretrained(
    "SinpxAI/Llama2-Chinese-7B-Chat-GGUF",
    model_file="llama2-chinese-7b-chat.Q4_K_M.gguf",
    model_type="llama",
    gpu_layers=0,
    hf=True
)

device="cuda:0"
model_llama = model_llama.to(device)
model_llama.eval()  # Set to eval mode

tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)
tokenizer_llama.pad_token = tokenizer_llama.eos_token
tokenizer_llama.padding_side = "right"

pipe_llama = pipeline(task="text-generation", model=model_llama, tokenizer=tokenizer_llama)

# Load fine-tuned bart model
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

model_bart = PeftModel.from_pretrained(base_model_reload, new_model)

model_bart = model_bart.to(device)
model_bart.eval()  # Set to eval mode

tokenizer_bart = BertTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer_bart.pad_token = tokenizer_bart.eos_token
tokenizer_bart.padding_side = "right"

pipe_bart = Text2TextGenerationPipeline(model=model_bart, tokenizer=tokenizer_bart)

# Load Knowledge Set
file_path = "./knowledge_set.txt"
data = []
# Open the file and parse each line from string to tuple
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():  # Ensure the line is not empty
            try:
                # Convert string representation of tuple to actual tuple
                tuple_data = ast.literal_eval(line.strip())
                data.append(tuple_data)
            except SyntaxError:
                print(f"Skipping malformed line: {line.strip()}")

# Load the data into a DataFrame
df = pd.DataFrame(data, columns=['Entity', 'Category', 'Answer'])

# Clean the knowledge set
def categoryClean(df , targetCategory):
        
        targetCategoryLen = len(targetCategory)

        # Count the "Category" value containing "targetCategory"
        comment_count = df[df['Category']== targetCategory].shape[0]

        filtered_data = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)]

        targetRow_ids = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)].index

        distinct_categories = df[df['Category'].str.contains(targetCategory)]['Category'].unique()
        
        ## update 
        df.loc[df['Category'].str.contains(targetCategory), 'Entity'] = df['Entity'] + df['Category'].str.split(targetCategory).str[0]
        df['Category'] = df['Category'].apply(lambda x: targetCategory if targetCategory in x else x)
        
        # Count the "Category" value containing "targetCategory"
        comment_count = df[df['Category'] == targetCategory].shape[0]

        filtered_data = df[df['Category'].str.contains(targetCategory) & (df['Category'].str.len() > targetCategoryLen)]

        distinct_categories = df[df['Category'].str.contains(targetCategory)]['Category'].unique()

        return df

cateList = ['评论', '主演','口碑','导演','类型','评分','国家地区','适合吃','特色菜','人均价格', '订单量','地址',]
for cat in cateList:
    df = categoryClean(df, cat)

# Knowledge Tree
G = nx.DiGraph()
# Add nodes and edges based on the DataFrame
for index, row in df.iterrows():
    entity_id = f"Entity: {row['Entity']}"
    category_id = f"Category: {row['Category']} ({row['Entity']})"
    answer_id = f"Answer: {row['Answer']} ({row['Category']})"

    # Ensure that nodes for each level (entity, category) are unique per entity-category pair
    if entity_id not in G:
        G.add_node(entity_id, type='Entity', name=row['Entity'])
    if category_id not in G:
        G.add_node(category_id, type='Category', name=row['Category'])
    
    # Answers can be multiple per category, so they are always added
    G.add_node(answer_id, type='Answer', content=row['Answer'])
    
    # Connect nodes hierarchically
    G.add_edge(entity_id, category_id)
    G.add_edge(category_id, answer_id)

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

    output = pipe_llama(
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
        inputs = tokenizer_bart.encode(full_input_without_markers, return_tensors="pt")
        outputs = model_bart.generate(input_ids=inputs.to(device), max_new_tokens=126)
        response_text = tokenizer_bart.decode(outputs[0], skip_special_tokens=True)
        return {"response": response_text}