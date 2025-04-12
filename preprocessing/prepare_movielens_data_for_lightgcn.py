import pandas as pd
import random


def extract_user_item_interactions_from_ratings():
    data = []

    with open('../data/Movielens/u.data', 'r') as file:
        for line in file:
            parts = line.strip().split('\t')  # split by tab
            if len(parts) == 4:  # ensure correct format
                data.append(parts)

    # Create DataFrame
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'rating', 'timestamp'])

    # Optional: convert data types
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    df['rating'] = df['rating'].astype(float)  # or int, if no decimal
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')  # if UNIX timestamp

    # Group by user_id and collect items and ratings
    grouped_by_user = df.groupby('user_id').apply(
        lambda x: pd.DataFrame({
            'items': [list(x['item_id'])],
            'ratings': [list(x['rating'])]
        })
    ).reset_index(level=1, drop=True).reset_index()

    # Open a file to write the output
    with open('../data/Movielens/user_items_interactions.txt', 'w') as f:
        for _, row in grouped_by_user.iterrows():
            user_id = row['user_id']
            item_ids = ' '.join(str(item) for item in row['items'])
            user_string = f"{user_id} {item_ids}"
            f.write(user_string + '\n')


def split_train_valid_test():
    # Prepare output files
    train_file = open('../data/Movielens/train.txt', 'w')
    valid_file = open('../data/Movielens/valid.txt', 'w')
    test_file = open('../data/Movielens/test.txt', 'w')

    # Read user_items.txt line by line
    with open('../data/Movielens/user_items_interactions.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            user_id = tokens[0]
            items = tokens[1:]

            # Shuffle items
            random.shuffle(items)
            n = len(items)

            train_split = int(n * 0.7)
            valid_split = int(n * 0.8)  # 70% + 10% = 80%

            train_items = items[:train_split]
            valid_items = items[train_split:valid_split]
            test_items = items[valid_split:]

            # Write to respective files
            if train_items:
                train_file.write(f"{user_id} {' '.join(train_items)}\n")
            if valid_items:
                valid_file.write(f"{user_id} {' '.join(valid_items)}\n")
            if test_items:
                test_file.write(f"{user_id} {' '.join(test_items)}\n")

    # Close all files
    train_file.close()
    valid_file.close()
    test_file.close()


if __name__ == '__main__':
    extract_user_item_interactions_from_ratings()
    split_train_valid_test()
