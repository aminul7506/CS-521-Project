

def prepare_item_information_data():
    # Define the 19 genre names in order
    genre_names = [
        "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
        "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    with (open('../data/Movielens/u.item', encoding='ISO-8859-1') as infile,
          open('../data/Movielens/movies.txt', 'w', encoding='utf-8') as outfile):
        for line in infile:
            parts = line.strip().split('|')
            if len(parts) >= 24:
                movie_id = parts[0]
                title = parts[1]
                release_date = parts[2]
                video_release_date = parts[3]
                imdb_url = parts[4]
                genre_flags = parts[5:24]

                # Generate genre string from flags
                genres = [name for name, flag in zip(genre_names, genre_flags) if flag == '1']
                genre_str = ', '.join(genres) if genres else 'None'

                # Tab-separated output
                output_line = f"{movie_id}|{title}|{release_date}|{imdb_url}|{genre_str}\n"
                outfile.write(output_line)


if __name__ == '__main__':
    prepare_item_information_data()