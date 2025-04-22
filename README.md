## CS-521-Project: Prompt-Based LLMs for Position Bias-Aware Reranking in Personalized Recommendations 

### LightGCN-pytorch

We use LightGCN for our base model for recommendation. We use the original code of LightGCN and
perform necessary modification for our purpose.

>SIGIR 2020. Xiangnan He, Kuan Deng ,Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang(2020). LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation, [Paper](https://dl.acm.org/doi/pdf/10.1145/3397271.3401063?casa_token=7ZAy9cglsYIAAAAA:1_Ed8H-_fbvkx5GpcyFyDh4khNChPmBXCehe5uSvu3pAjdKI3XxKtRtX07rrwqOo-1vwImHK8acr).

### Enviroment Requirement

`pip install -r requirements.txt`

### Dataset

We provide the [Movielens-100k](https://grouplens.org/datasets/movielens) dataset. We processed the data 
for LightGCN format and post-process for LLM format. The dataset is provided in data/Movielens/ folder. 
Belows are the extracted pre-procssed files from the Movielens-100k dataset designed for our purporse. 
The original dataset descriptions of the Movielens-100k can be found in data/Movielens/README.

1. `movies.txt`: Each line is formatted in this format - movie id | title | release date | imdb_url | genres_str_comma_seperated.
2. `users.txt`: Each line is formatted in this format - user id | age | gender | occupation | zip code
3. `user_items_interactions.txt`: Each line begins with the user ID, followed by the item IDs to which the user has given ratings.
4. `train.txt, valid.txt, test.txt`: We prepare these data from `user_items_interactions.txt` file. For each user, we split 70%, 10%, and 20% items for train, test, and valid.
5. `topk_predict.txt`: Each line begins with the user ID, followed by the top-k predicted item IDs.

### An example to run a 3-layer LightGCN with Movielens dataset.

```cd code && ./run.sh Movielens lgn 3 1000```


### Results for LightGCN with Movielens dataset with layer 3
*all metrics is under top-20*

(*for seed=2020*)

* Movielens-100k:

|             | Recall | ndcg    | precision |
| ----------- | ------------------------- |---------|-----------|
| **layer=3** | 0.41340                | 0.46346 | 0.29804   |

### Generate top-k with LLM prompt
1. To generate the top-k predicted items for each user using an LLM, run the following script: 
```python
python code/llm_prompting_predict_top-k.py
```
2. To generate the top-k predicted items with randomized interaction history for each user using an LLM, run the following script: 
```python
python code/llm_prompting_predict_top-k_randomized_history.py
```
3. To generate the top-k predicted items with reduced postion bias for each user using an LLM, run the following script: 
```python
python code/llm_prompting_predict_top-k_reduce_position_bias.py
```
