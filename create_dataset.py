import pandas as pd


def compress_basic_data(filename):
    """
    A function to choose all the films with the value "movie"
    """
    basic_data = open(filename, 'r', encoding="utf8").readlines()
    movie_basic_data = open("title_movies_basics.tsv", 'w', encoding='utf8')
    movie_basic_data.write(basic_data[0])
    for line in basic_data:
        data = line.split('\t')
        if data[1] == 'movie':
            movie_basic_data.write(line)
    movie_basic_data.close()

data = open('Data/Box_office_directors.csv', encoding='utf8').readlines()
print(len(data))
print("reading basics data...")
basics_data = pd.read_table('Data/title_movies_basics.tsv')
# Removing all the movies without a start year
basics_data = basics_data[basics_data.startYear != '\\N']
# Converting all startYears to ints
basics_data['startYear'] = basics_data['startYear'].astype('int')
# Sorting the startYears with the most recent one first
basics_data = basics_data.sort_values(by='startYear', ascending=False)
# Picking the movies made between 1990 and 2018
basics_data = basics_data[basics_data.startYear >= 1990]
basics_data = basics_data[basics_data.startYear <= 2018]
# Changing the index from numbers to tconst
basics_data.index = basics_data.tconst

print("reading crew data...")
crew_data = pd.read_table('Data/title_crew.tsv')

print("reading ratings data...")
ratings_data = pd.read_table('Data/title_ratings.tsv')

# Set indexes to tconst
crew_data.index = crew_data.tconst
ratings_data.index = ratings_data.tconst

# Labels from each file to be included in the new file
basics_labels = ['primaryTitle', 'genres']
crew_labels = ['directors', 'writers']
rating_labels = ['averageRating']

# Concatenates the columns and keeps only the columns
#movies_dataset = pd.concat([basics_data[basics_labels], crew_data[crew_labels], ratings_data[rating_labels]],
#                        axis=1, join='inner')

#movies_dataset.to_csv('movies_dataset.csv')

# Open box office dataset
movies_dataset = pd.read_csv('data/Movies_data_Merged_12-17.csv', sep=';')
print(len(movies_dataset))

# Make Box_Office to int
movies_dataset['Box_Office'] = movies_dataset['Box_Office'].apply(lambda x: int(x.replace(',', '')))
# Make genres to list
movies_dataset['genres'] = movies_dataset['genres'].apply(lambda x: x.split(','))
# Make directors to list
movies_dataset['directors'] = movies_dataset['directors'].apply(lambda x: x.split(','))
# Make writers to list
movies_dataset['writers'] = movies_dataset['writers'].apply(lambda x: x.split(','))
# Make directorsNames to list
movies_dataset['directorsNames'] = movies_dataset['directorsNames'].apply(lambda x: x.split(','))
# Convert rating to float
movies_dataset['averageRating'] = pd.to_numeric(movies_dataset['averageRating'])
# Convert Year(box) to float
movies_dataset['Year(box)'] = pd.to_numeric(movies_dataset['Year(box)'])
# Convert Year to numeric
movies_dataset['Year'] = pd.to_numeric(movies_dataset['Year'])
# Convert votes to numeric
movies_dataset['votes'] = pd.to_numeric(movies_dataset['votes'])

movies_dataset.to_csv('final_movies_dataset.csv')
