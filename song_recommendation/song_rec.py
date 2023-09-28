import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

def user_prompt():
  song_name = input("Please enter song name: ")
  artist_name = input("Please enter artist name: ")
  show_analysis = input("Show graph analysis? (Y or N): ").lower().strip() == 'y'

  return song_name, artist_name, show_analysis

def prepare_data(input_data, show_analysis):


  if (show_analysis):
    print("Printing first 5 rows of dataframe...")
    print(input_data.head())

    # Diagnostics for data preparation
    print("\nPrinting dimensions of dataframe...")
    print(input_data.shape)

    print("\nPrinting general information about the dataset...")
    print(input_data.info())

    print("\nChecking for any null values in columns of dataset...")
    print(input_data.isnull().sum())

  #print("\nDeleting rows with null column values...")
  #input_data.dropna(inplace = True)
  #print(input_data.isnull().sum())

  # Removing columns that are irrelevant for recommender system
  cleaned_data = input_data.drop(['Unnamed: 0', 'lyrics', 'len', 'dating', 'violence', 'world/life', 'night/time', 
  'shake the audience', 'family/gospel', 'romantic', 'communication', 'obscene', 'music', 'movement/places',
  'light/visual perceptions', 'family/spiritual', 'like/girls', 'sadness', 'feelings', 'topic', 'age'], axis = 1)

  if (show_analysis):
    print ("\nRemoving unncessary columns...")
    print(cleaned_data.head())
  
  return cleaned_data

def exp_data_analysis(data):

  # Exploratory Data Analysis
  numerical_columns = data.drop(['artist_name', 'genre', 'track_name', 'release_date'], axis = 1)
  model = TSNE(n_components = 2, random_state = 0)
  tsne_data = model.fit_transform(numerical_columns.head(500))
  plt.figure(figsize = (7,7))
  plt.scatter(tsne_data[:,0], tsne_data[:,1])

  # To check for non unique songs (different versions of same song)
  #data['track_name'].nunique()
  #print(data.shape) ==> (x, (y, z)) if there are non unique songs

  # Visualizing number of songs released per year
  plt.figure(figsize =  (10, 5))
  sb.histplot(data=data, x ='release_date')

  # Insights for numerical columns
  plt.subplots(figsize = (15, 5))
  for i, col in enumerate(numerical_columns):
      plt.subplot(2, 5, i + 1)
      sb.distplot(data[col])
  plt.tight_layout()

  plt.show()

  

def get_similarities(song_name, artist_name, data):
   
  song_vectorizer = CountVectorizer()
  song_vectorizer.fit(data['genre'])

  # Getting vector for the input song.
  text_array1 = song_vectorizer.transform(data[data['track_name']==song_name][data['artist_name']==artist_name]['genre']).toarray()
  num_array1 = data[data['track_name']==song_name][data['artist_name']==artist_name].select_dtypes(include=np.number).to_numpy()
   
  # We will store similarity for each row of the dataset.
  sim = []
  for idx, row in data.iterrows():
    track_name = row['track_name']
     
    # Getting vector for current song.
    text_array2 = song_vectorizer.transform(data[data['track_name']==track_name]['genre']).toarray()
    num_array2 = data[data['track_name']==track_name].select_dtypes(include=np.number).to_numpy()
 
    # Calculating similarities for text as well as numeric features
    text_sim = cosine_similarity(text_array1, text_array2)[0][0]
    num_sim = cosine_similarity(num_array1, num_array2)[0][0]
    sim.append(text_sim + num_sim)
     
  return sim

def recommend_songs(song_name, artist_name, data):
  # Base case
  if tracks[tracks['track_name'] == song_name][tracks['artist_name']==artist_name].shape[0] == 0:
    print('This song is either not so popular or you\
    have entered invalid_name.\n Some songs you may like:\n')
     
    for song in data.sample(n=5)['track_name'].values:
      print(song)
    return
   
  data['similarity_factor'] = get_similarities(song_name, artist_name, data)
 
  data.sort_values(by=['similarity_factor', 'energy'],
                   ascending = [False, False],
                   inplace=True)
   
  print(data[['track_name', 'artist_name']][0:7])

if __name__ == '__main__':

  tracks = pd.read_csv('tcc_ceds_music.csv')
  song_name, artist_name, show_analysis = user_prompt()
  cleaned_tracks = prepare_data(tracks, show_analysis)

  if (show_analysis):
    exp_data_analysis(cleaned_tracks)

  print("Recommending songs similar to " + song_name + " by " + artist_name + "...")
  recommend_songs(song_name, artist_name, cleaned_tracks)
