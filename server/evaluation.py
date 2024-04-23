#
import pandas as pd
from datasets import load_dataset
import re
from rouge import Rouge

#
from models import load_llama_model, load_bart_model, query_llama
from knowledgeTree import load_knowledge_set, build_tree
from utils import retrieve_answers

# Hugging Face Access Token
hf_access_token = "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"

# Load Llama2 Pipeline
llama_pipe = load_llama_model(hf_access_token)

# Load Fine-tuned BART
bart_model, bart_tokenizer =  load_bart_model

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

#query_llama(G, userinput)

# Evaluation Dataset Preprocessing
# Load test.txt for evaluation
dataset = load_dataset(
    'json',
    data_files={
        'test':
        '/kaggle/input/durecdial/test.txt',
    }
)

# Generate result.txt
## Preprocess test.txt Dataset
conversation_list = dataset['test']['conversation']
conversation_list

# Format each turn in the context
formatted_context = []

for conversations in conversation_list:
    
    ## Label the last user inquiry for Llama2(Knowledge Tree) inquiry approach
    conversations[-1] = conversations[-1].split(':')[0] + ':' + '##@' + conversations[-1].split(':')[1] + '@##'
    
    ## Formatted the whole data for BART inquiry approach
    formatted_conversation = ''.join(conv.replace('user:', '\nInstruction:\n').replace('bot:','\nResponse:\n').replace(' ','') for conv in conversations)
    
    ## Append into a list 
    formatted_context.append(formatted_conversation.lstrip('\n'))
    
#
def generate_text(user_input):
    # Retrieve the latest user input
    match = re.search(r'##@(.*?)@##', user_input)
    
    if match:
        # Extracts text found between the markers
        query = match.group(1)
        full_input_without_markers = re.sub(r'##@.*?@##', query, user_input)
    else:
        query = user_input
        full_input_without_markers = user_input
    
    # Attempt to retrieve from the knowledge set for Llama2
    answers, require_llama = retrieve_answers(G, query)

    # If answers are found in the graph, and further processing with Llama2 is required
    if answers and require_llama:
        return query_llama(query, answers)
    elif answers and not require_llama:
        return answers
    else:
        # If no answers are found, use BART model to generate an answer from the full context without markers
        inputs = bart_tokenizer.encode(full_input_without_markers, return_tensors="pt")
        outputs = bart_model.generate(input_ids=inputs.to("cuda:0"), max_new_tokens=126)
        response_text = bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response_text
    
# Evaluation by predicting the response of each conversation history
# Generate Response on the test.txt dataset
total_result = []

for con in formatted_context:
    con_result = generate_text(con)
    total_result.append(con_result)
    
# Create a new list to store the results
# total_result contains list elements, needs to convert into string for Evaluation
modified_total_result = []

# Iterate through candidate_result and convert inner lists to strings
for item in total_result:
    if isinstance(item, list):
        modified_total_result.append(' '.join(item).replace(' ',''))
    else:
        modified_total_result.append(item.replace(' ',''))
        
# Create a DataFrame to store prediction results for each record.
df_response = pd.DataFrame(modified_total_result, columns=['Response'])

## Generate result.txt for submission
df_response['Response'].to_csv('result.txt', index=False, header=False)

# Evaluation Metrics
## Evaluate dataset, which is a conversation before the last bot response
eval_test_set = []

## Evaluate dataset reference, which is the last bot response in a conversation
eval_reference = []

## Data Preprocessing for spliting evaluation conversation and reference
for conversations in conversation_list:
    eval_data = conversations[:-2]
    reference = conversations[-2]
    
    ## Label the last user inquiry for Llama2(Knowledge Tree) inquiry approach
    eval_data[-1] = eval_data[-1].split(':')[0] + ':' + '##@' + eval_data[-1].split(':')[1] + '@##'
    
    ## Formatted the whole data for BART inquiry approach
    formatted_eval_data = ''.join(eval_data_conv.replace('user:', '\nInstruction:\n').replace('bot:','\nResponse:\n').replace(' ','') for eval_data_conv in eval_data)
    
    ## Append into eval_test_set list 
    eval_test_set.append(formatted_eval_data.lstrip('\n'))
    
    ## Append into eval_reference list 
    eval_reference.append(reference)
    
## Check that all references are responded to by "bot"
sum_bot_res = 0

for reference in eval_reference:
    if reference.__contains__('bot: '):
        sum_bot_res = sum_bot_res + 1
    
print('bot count' ,sum_bot_res)

## Remove irrelevant string in eval_reference
eval_reference = [s.replace("bot: ", "").replace(" ", "") for s in eval_reference]

## Generate candidate on the eval_test_set(test.txt)
candidate_result = []

for con in eval_test_set:
    con_result = generate_text(con)
    candidate_result.append(con_result)
    
# Create a new list to store the modified candidate results
# candidate_result contains list element, needs to convert into string for Evaluation

modified_candidate_result = []

# Iterate through candidate_result and convert inner lists to strings
for item in candidate_result:
    if isinstance(item, list):
        modified_candidate_result.append(' '.join(item).replace(' ',''))
    else:
        modified_candidate_result.append(item.replace(' ',''))
        
## Store into dataframe
df_eval_reference = pd.DataFrame(eval_reference, columns=['Reference'])

## Store into dataframe
df_eval_candidate = pd.DataFrame(modified_candidate_result, columns=['Candidate'])

## Generate Candidate csv
df_eval_candidate['Candidate'].to_csv('eval_candidate.txt', index=False, header=False)

# Calculate BLEU individual scores
bleu_individual_1_gram = [sentence_bleu(ref, cand, weights=(1, 0, 0, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_individual_2_gram = [sentence_bleu(ref, cand, weights=(0, 1, 0, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_individual_3_gram = [sentence_bleu(ref, cand, weights=(0, 0, 1, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_individual_4_gram = [sentence_bleu(ref, cand, weights=(0, 0, 0, 1)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]

## Get bleu individual avg for all record
bleu_individual_1_gram_avg = sum(bleu_individual_1_gram) / len(bleu_individual_1_gram)
bleu_individual_2_gram_avg = sum(bleu_individual_2_gram) / len(bleu_individual_2_gram)
bleu_individual_3_gram_avg = sum(bleu_individual_3_gram) / len(bleu_individual_3_gram)
bleu_individual_4_gram_avg = sum(bleu_individual_4_gram) / len(bleu_individual_4_gram)

# Print bleu reslut
print("bleu_individual_1_gram_avg: ", bleu_individual_1_gram_avg)
print("bleu_individual_2_gram_avg: ", bleu_individual_2_gram_avg)
print("bleu_individual_3_gram_avg: ", bleu_individual_3_gram_avg)
print("bleu_individual_4_gram_avg: ", bleu_individual_4_gram_avg)

# bleu individual avge result
bleu_individual_avg = (bleu_individual_1_gram_avg+bleu_individual_2_gram_avg+bleu_individual_3_gram_avg+bleu_individual_4_gram_avg) / 4
print("bleu_individual_avg: ", bleu_individual_avg)

# Calculate BLEU cumulative scores
bleu_cumulative_1_gram = [sentence_bleu(ref, cand, weights=(1, 0, 0, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_cumulative_2_gram = [sentence_bleu(ref, cand, weights=(0.5, 0.5, 0, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_cumulative_3_gram = [sentence_bleu(ref, cand, weights=(0.33, 0.33, 0.33, 0)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
bleu_cumulative_4_gram = [sentence_bleu(ref, cand, weights=(0.25, 0.25, 0.25, 0.25)) for ref, cand in  zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]

## Get bleu avg for all record
bleu_cumulative_1_gram_avg = sum(bleu_cumulative_1_gram) / len(bleu_cumulative_1_gram)
bleu_cumulative_2_gram_avg = sum(bleu_cumulative_2_gram) / len(bleu_cumulative_2_gram)
bleu_cumulative_3_gram_avg = sum(bleu_cumulative_3_gram) / len(bleu_cumulative_3_gram)
bleu_cumulative_4_gram_avg = sum(bleu_cumulative_4_gram) / len(bleu_cumulative_4_gram)

# Print bleu reslut
print("bleu_cumulative_1_gram_avg: ", bleu_cumulative_1_gram_avg)
print("bleu_cumulative_2_gram_avg: ", bleu_cumulative_2_gram_avg)
print("bleu_cumulative_3_gram_avg: ", bleu_cumulative_3_gram_avg)
print("bleu_cumulative_4_gram_avg: ", bleu_cumulative_4_gram_avg)

# bleu avge result
bleu_avg = (bleu_cumulative_1_gram_avg+bleu_cumulative_2_gram_avg+bleu_cumulative_3_gram_avg+bleu_cumulative_4_gram_avg) / 4
print("bleu_avg: ", bleu_avg)

## Evaluation using rouge
#!pip install rouge-chinese

## Split by char
rouge = Rouge()

## rouge.get_scores(hyps=' '.join(df_eval_candidate['Candidate'][3]) , refs=' '.join(df_eval_reference['Reference'][3]) )

# Calculate rouge F1 scores
rouge_1_f1 = [rouge.get_scores(hyps=' '.join(cand) , refs=' '.join(ref) )[0]['rouge-1']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
rouge_2_f1 = [rouge.get_scores(hyps=' '.join(cand) , refs=' '.join(ref) )[0]['rouge-2']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
rouge_l_f1 = [rouge.get_scores(hyps=' '.join(cand) , refs=' '.join(ref) )[0]['rouge-l']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]

## Get rouge F1  avg for all record
rouge_1_f1_avg = sum(rouge_1_f1) / len(rouge_1_f1)
rouge_2_f1_avg = sum(rouge_2_f1) / len(rouge_2_f1)
rouge_l_f1_avg = sum(rouge_l_f1) / len(rouge_l_f1)

# Print rouge F1  reslut
print("rouge_1_f1_avg: ", rouge_1_f1_avg)
print("rouge_2_f1_avg: ", rouge_2_f1_avg)
print("rouge_l_f1_avg: ", rouge_l_f1_avg)

## Split by word

from rouge import Rouge
import jieba

rouge = Rouge()

## rouge.get_scores(hyps=' '.join(df_eval_candidate['Candidate'][3]) , refs=' '.join(df_eval_reference['Reference'][3]) )

# Calculate rouge F1 scores
rouge_1_word_f1 = [rouge.get_scores(hyps=' '.join(jieba.cut(cand, HMM=False)) , refs=' '.join(jieba.cut(ref, HMM=False)) )[0]['rouge-1']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
rouge_2_word_f1 = [rouge.get_scores(hyps=' '.join(jieba.cut(cand, HMM=False)) , refs=' '.join(jieba.cut(ref, HMM=False)) )[0]['rouge-2']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]
rouge_l_word_f1 = [rouge.get_scores(hyps=' '.join(jieba.cut(cand, HMM=False)) , refs=' '.join(jieba.cut(ref, HMM=False)) )[0]['rouge-l']['f'] for ref, cand in zip(df_eval_reference['Reference'], df_eval_candidate['Candidate'] )]

         
## Get rouge F1  avg for all record
rouge_1_word_f1_avg = sum(rouge_1_word_f1) / len(rouge_1_word_f1)
rouge_2_word_f1_avg = sum(rouge_2_word_f1) / len(rouge_2_word_f1)
rouge_l_word_f1_avg = sum(rouge_l_word_f1) / len(rouge_l_word_f1)

# Print rouge F1  reslut
print("rouge_1_word_f1_avg: ", rouge_1_word_f1_avg)
print("rouge_2_word_f1_avg: ", rouge_2_word_f1_avg)
print("rouge_l_word_f1_avg: ", rouge_l_word_f1_avg)


## ROUGE and BLEU Metric
# Print bleu reslut
print("bleu_individual_1_gram_avg: ", bleu_individual_1_gram_avg)
print("bleu_individual_2_gram_avg: ", bleu_individual_2_gram_avg)
print("bleu_individual_3_gram_avg: ", bleu_individual_3_gram_avg)
print("bleu_individual_4_gram_avg: ", bleu_individual_4_gram_avg)

# Print bleu reslut
print("bleu_cumulative_1_gram_avg: ", bleu_cumulative_1_gram_avg)
print("bleu_cumulative_2_gram_avg: ", bleu_cumulative_2_gram_avg)
print("bleu_cumulative_3_gram_avg: ", bleu_cumulative_3_gram_avg)
print("bleu_cumulative_4_gram_avg: ", bleu_cumulative_4_gram_avg)

# Print bleu reslut
print("rouge_1_f1_avg (Char): ", rouge_1_f1_avg)
print("rouge_2_f1_avg (Char): ", rouge_2_f1_avg)
print("rouge_l_f1_avg (Char): ", rouge_l_f1_avg)

# Print bleu reslut
print("rouge_1_word_f1_avg: ", rouge_1_word_f1_avg)
print("rouge_2_word_f1_avg: ", rouge_2_word_f1_avg)
print("rouge_l_word_f1_avg: ", rouge_l_word_f1_avg)