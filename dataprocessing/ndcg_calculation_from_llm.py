import numpy as np


def dcg_at_k(predicted, ground_truth, k):
    dcg = 0.0
    for i, item in enumerate(predicted[:k]):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)  # rank position starts at 1
    return dcg


def ndcg_at_k(predicted, ground_truth, k):
    dcg = dcg_at_k(predicted, ground_truth, k)
    ideal_dcg = dcg_at_k(ground_truth[:k], ground_truth, k)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def read_file_for_ranked_items(filename):
    user_items = {}

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            user_id = parts[0]
            item_ids = list(map(int, parts[1:]))
            user_items[user_id] = item_ids
            print(item_ids)
    # print(user_items)
    return user_items


def calculate_ndcg_at_k(true_items, predicted_items, k):
    ndcg = 0
    for user_id, item_list_true in true_items.items():
        # print(f"User {user_id}: Items = {item_list_true}")
        ndcg += ndcg_at_k(predicted_items[user_id], item_list_true, k)

    print("NDCG@" + str(k) + ": " + str (ndcg / len(true_items)))
    return ndcg / len(true_items)


if __name__ == '__main__':
    true_lists = read_file_for_ranked_items("../data/Movielens/test.txt")
    llm_predicted_lists = read_file_for_ranked_items("../data/Movielens/topk_predict_llm.txt")
    # llm_predicted_lists = read_file_for_ranked_items("../data/Movielens/topk_predict_llm_with_randomized_interactions.txt")

    results = {}
    ndcgs = []
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 1))
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 3))
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 5))
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 10))
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 15))
    ndcgs.append(calculate_ndcg_at_k(true_lists, llm_predicted_lists, 20))

    results["ndcg"] = np.array(ndcgs)
    print(results)

    # Write to a txt file
    with open('../data/Movielens/results_llm.txt', 'w') as f:
        f.write(f"{results}\n")