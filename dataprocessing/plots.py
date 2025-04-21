import matplotlib.pyplot as plt
import numpy as np


def read_ndcg_scores(filename):
    with open(filename, 'r') as f:
        content = f.read()
        data = eval(content, {"array": np.array})
        ndcg_scores = data['ndcg']
        ndcg_modified = np.delete(ndcg_scores, 1)
        return ndcg_modified


def plot_figure(research_question):
    # Define k values
    k_values = [1, 5, 10, 15, 20]

    ndcg_lightgcn = read_ndcg_scores("../data/Movielens/results_lgn.txt")
    ndcg_lm = read_ndcg_scores("../data/Movielens/results_llm.txt")
    if research_question == 2:
        ndcg_lm_with_randomization = read_ndcg_scores("../data/Movielens/results_llm_with_randomized_interactions.txt")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, ndcg_lightgcn, marker='s', label='LightGCN')
    plt.plot(k_values, ndcg_lm, marker='o', label='GPT‑4o')
    if research_question == 2:
        plt.plot(k_values, ndcg_lm_with_randomization, marker='p', label='GPT‑4o (Rand)')
    tick_labels = ['1', '5', '10', '15', '20']
    tick_positions = [1, 5, 10, 15, 20]
    plt.xticks(tick_positions, tick_labels)

    plt.xlabel('k')
    plt.ylabel('NDCG@k')
    plt.title('')
    # plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig("../data/Movielens/RQ%d.png" % research_question,
                format="png", dpi=500, bbox_inches='tight')


if __name__ == '__main__':
    RQ = 2
    plot_figure(RQ)