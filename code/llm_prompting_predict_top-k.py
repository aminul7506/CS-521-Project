import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
import string
import warnings
import re # helps you filter urls
from scipy import sparse
from IPython.display import display, Latex, Markdown
warnings.filterwarnings('ignore')
import openai
import time
from openai import OpenAI
import time
from tqdm import tqdm
import pandas as pd
np.random.seed(4200)

dataset_name = "Movielens"
path_prefix = "../data/" + dataset_name + "/"
# Read the files
df_users = pd.read_csv(path_prefix + "users.txt", sep='|', header=None)
df_items = pd.read_csv(path_prefix + "movies.txt", sep='|', header=None)
# Assign column names
# df_users.columns = ['UserID', 'Age', 'Gender', 'Occupation', 'ZipCode']
# df_items.columns = ['ItemID', 'Name', 'Date', 'url', 'genre']

#create a dictionary interactions where key is the user_id and value is the previously interacted item list
interactions = {}
with open(path_prefix + "train.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        user_id = int(parts[0])
        item_ids = list(map(int, parts[1:]))
        interactions[user_id] = item_ids

#create a dictionary top_k where key is the user_id and value is the top-k predicted item list
top_k = {}
with open(path_prefix + "topk_predict.txt", "r") as file:
    for line in file:
        parts = line.strip().split()
        user_id = int(parts[0])
        item_ids = list(map(int, parts[1:]))
        top_k[user_id] = item_ids


output_file_name = path_prefix + "top-k_predict_llm.txt"
output_file = open(output_file_name, "w")


#Setup OpenAI/ChatGPT API
# Visit https://platform.openai.com/docs/overview and create an OpenAI account.
# Click on the upper right corner icon once logged in and select "View API Keys" in OpenAI. 
# Generate keys and copy/paste them into a file named 'openai-key.txt' in the current directory. 
# (Note: we are taking these steps to keep your keys private and excluded from your submitted PDF/files.)
with open('openai-key.txt', 'r') as file:
    openai_key = file.read().rstrip()
#  set up a client for the OpenAI API with the api key.    
client = OpenAI(api_key=openai_key)

# define a function for calling OpenAI/ChatGPT API endpoints for chat completions.
def chatGPT(client, input_string, prompt="You are a helpful assistant.", model="gpt-4.1-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
              "role": "system",
              "content": prompt
            },
            {
              "role": "user",
              "content": input_string
            }
        ],
        temperature=0.7,
        max_tokens=2048,
        top_p=1
    )
    return response.choices[0].message.content
    

# Define a method to rerank items as per the prompt which are passing via four parameters
# count = 0
def LLM_reranking(history_prefix, history_items, candidate_prefix, cadidate_items):
  global count
  history_index = 0
  candidate_index = 0
  prompt = history_prefix + '\n'
  prompt += '{'
  for history in history_items:
    history_index += 1
    prompt = prompt + history + ". "
  prompt += '}. \n'
  prompt +=  candidate_prefix + '\n'
  prompt += '{'
  for candidate in cadidate_items:
    candidate_index += 1
    prompt = prompt + candidate + ". "
  prompt += '}.'
  # print(prompt)

  LLM_response = chatGPT(client, prompt)
  # print(LLM_response)
  LLM_item_ranking = list(map(int, LLM_response.split(",")))
  # count += 1
  # print(count, '\n')
  return LLM_item_ranking
  
  
# Depending on the LLM output, the following loop may create exception. 
# In that case, the loop needs to be resumed from user i where it got stuck.  
for i in tqdm(range(0, df_users.shape[0])):
  history_prefix = None
  history_items = None
  candidate_prefix = None
  candidate_items = None
  age = df_users.iloc[i][1]
  gender = df_users.iloc[i][2]
  occupation = df_users.iloc[i][3]
  zipcode = df_users.iloc[i][4]
  if gender == 'M':
    gender = 'male'
  else:
    gender = 'female'
  history_prefix = "A user with " + " age: " + str(age) + ", gender: " + gender + ", occupation: " + occupation + ", zipcode: " + zipcode + " has interacted with the following items (in orders) previously: "
  history_items = []
  item_list = interactions[i+1]
  for j in item_list:
    history_items.append(" item_ID: " + str(df_items.iloc[j-1][0]) + ", Name: "  + df_items.iloc[j-1][1] + ", Genre: " + df_items.iloc[j-1][4])

  top_k_list = top_k[i+1]
  # print("original list: ", len(top_k_list), "items ", top_k_list)
  LLM_item_ranking = []
  while len(LLM_item_ranking) == 0 or len(list(set(LLM_item_ranking) | set(top_k_list))) != len(top_k_list):
    candidate_prefix = "From the items listed below, rank them in order of relevance from highest to lowest for the user based on the users previous history given to you. Output the ranked items as comma separated item_ID without explaining the reason or include any other words. Make sure the output contains all the unique item_IDs given to you and do not skip any item. Candidates: "
    candidate_items = []
    for j in top_k_list:
      candidate_items.append("item ID: " + str(j) + ", " + df_items.iloc[j-1][1] + ", Genre: " + df_items.iloc[j-1][4])
    LLM_item_ranking = LLM_reranking(history_prefix, history_items, candidate_prefix, candidate_items)
    # print(len(LLM_item_ranking), "items ", LLM_item_ranking)
    time.sleep(0.001)
  with open(output_file_name, "a") as f:
    f.write(f"{i+1} " + " ".join(str(m) for m in LLM_item_ranking) + "\n")
  # time.sleep(0.001)
  # print("original list: ", len(top_k_list), "items ", top_k_list)
  # print("reranked list: ", len(LLM_item_ranking), "items ", LLM_item_ranking)
