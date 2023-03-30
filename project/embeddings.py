# %%
# %pip install openai


import os
os.system('pip install matplotlib')
os.system('pip install plotly')
os.system('pip install scipy')
os.system('pip install sklearn')
os.system('pip install scikit-learn')

# %%
import pandas as pd
import pickle

# %%
import os
import openai
openai.organization = "org-9bUDqwqHW2Peg4u47Psf9uUo"
openai.Model.list()

# %%
from typing import List
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    tsne_components_from_embeddings,
    chart_from_components,
    indices_of_nearest_neighbors_from_distances,
)

# constants
EMBEDDING_MODEL = "text-embedding-ada-002"


# %%
outline_sections = [
    "It took me eighteen years to realize what an extraordinary influence my mother has been on my life.",
    "Her mother's enthusiasm for learning.",
    "Learning through travel by using the example of a trip to Greece.",
    "While I treasure the various worlds my mother has opened to me abroad, my life has been equally transformed by what she has shown me just two miles from my house.",
    "Her mother's dedication to the community.",
    "Her multiple volunteer activities such as helping at the local soup kitchen.",
    "Everything that my mother has ever done has been overshadowed by the thought behind it.",
    "She has enriched my life with her passion for learning, and changed it with her devotion to humanity.",
    "Next year, I will find a new home miles away. However, my mother will always be by my side."
]

# %%
texts = [
"It took me eighteen years to realize what an extraordinary influence my mother has been on my life.",
"She's the kind of person who has thoughtful discussions about which artist she would most want to have her portrait painted by (Sargent), the kind of mother who always has time for her four children, and the kind of community leader who has a seat on the board of every major project to assist Washington's impoverished citizens.",
"Growing up with such a strong role model, I developed many of her enthusiasms.I not only came to love the excitement of learning simply for the sake of knowing something new, but I also came to understand the idea of giving back to the community in exchange for a new sense of life, love, and spirit.",
"My mother's enthusiasm for learning is most apparent in travel.",
"Despite the fact that we were traveling with fourteen-month-old twins, we managed to be at each ruin when the site opened at sunrise.",
"I vividly remember standing in an empty amphitheatre pretending to be an ancient tragedian, picking out my favorite sculpture in the Acropolis museum, and inserting our family into modified tales of the battle at Troy.",
"Eight years and half a dozen passport stamps later I have come to value what I have learned on these journes about global history, politics and culture, as well as my family and myself.",
"I was nine years old when my family visited Greece. Every night for three weeks before the trip, my older brother Peter and I sat with my mother on her bed reading Greek myths and taking notes on the Greek Gods.",
'''While I treasure the various worlds my mother has opened to me abroad, my life has been equally transformed by what she has shown me just two miles from my house.''',
'''As a ten year old, I often accompanied my mother to (name deleted), a local soup kitchen and children's center.''',
"While she attended meetings, I helped with the Summer Program by chasing children around the building and performing magic tricks.",
'''Having finally perfected the "floating paintbrush" trick, I began work as a full time volunteer with the five and six year old children last June.''',
"It is here that I met Jane Doe, an exceptionally strong girl with a vigor that is contagious.",
"At the end of the summer, I decided to continue my work at (name deleted) as Jane's tutor.Although the position is often difficult, the personal rewards are beyond articulation.",
"In the seven years since I first walked through the doors of (name deleted), I have learned not only the idea of giving to others, but also of deriving from them a sense of spirit.",
"Everything that my mother has ever done has been overshadowed by the thought behind it.",
"While the raw experiences I have had at home and abroad have been spectacular, I have learned to truly value them by watching my mother.",
"She has enriched my life with her passion for learning, and changed it with her devotion to humanity.",
"In her endless love of everything and everyone she is touched by, I have seen a hope and life that is truly exceptional.",
"Next year, I will find a new home miles away. However, my mother will always be by my side."
]

# %%
def get_distances_from_query_list(query_list, texts):
    list_emb = [get_embedding(text) for text in texts]
    distances = []
    for query in query_list:
        query_emb = get_embedding(query)
        distances.append(distances_from_embeddings(query_emb, list_emb))
    return distances

# %%
distances = get_distances_from_query_list(outline_sections, texts)
distances


# %%
def recommendations_from_strings(
   strings: List[str],
   index_of_source_string: int,
   model="text-embedding-ada-002",
) -> List[int]:

   # get embeddings for all strings
   embeddings = [get_embedding(string) for string in strings]
   
   # get the embedding of the source string
   query_embedding = embeddings[index_of_source_string]
   
   # get distances between the source embedding and other embeddings (function from embeddings_utils.py)
   distances = distances_from_embeddings(query_embedding, embeddings, distance_metric="cosine")
   
   # get indices of nearest neighbors (function from embeddings_utils.py)
   indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)
   return indices_of_nearest_neighbors

# %%
query_emb = get_embedding(query)

# %%
list_emb = [get_embedding(text) for text in texts]

# %%
distances_from_embeddings(query_emb, list_emb)


