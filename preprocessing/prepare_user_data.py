
def prepare_user_demographics_data():
    # Read from u.user (or your file) and write to a new space-separated file
    with open('../data/Movielens/u.user', 'r') as infile, open('../data/Movielens/users.txt', 'w') as outfile:
        for line in infile:
            fields = line.strip().split('|')  # Split by tab
            if len(fields) == 5:  # Ensure correct format
                outfile.write('|'.join(fields) + '\n')


if __name__ == '__main__':
    prepare_user_demographics_data()