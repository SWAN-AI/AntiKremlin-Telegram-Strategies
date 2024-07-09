from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import numpy as np
import pandas as pd
import math
from umap import UMAP
import umap
from hdbscan import HDBSCAN 
from googletrans import Translator
translator = Translator()
import zstandard
from bertopic.representation import MaximalMarginalRelevance
import pandas as pd
import re
import numpy as np
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

df = pd.read_csv("antiKremlin_dataset.csv")

df = df[df['processed_message'].notnull()]
df.reset_index(drop=True, inplace=True)

def compute_coherence_values(start, limit, step, processed_message_cleaned):
    coherence_values = []
    
    representation_model = MaximalMarginalRelevance(diversity=0.8)
    umap_models = UMAP(n_neighbors=15, 
                  n_components=5, 
                  min_dist=0.0, 
                  metric='cosine', 
                  random_state=101)

    sentence_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    

    for n_topics in range(start, limit, step):
        topic_model = BERTopic(umap_model=umap_models,  
                               calculate_probabilities=False, 
                               embedding_model=sentence_model,
                               nr_topics = n_topics)
        
        docs = processed_message_cleaned.tolist()
        
        topics, probabilities = topic_model.fit_transform(processed_message_cleaned)

        # Preprocess Documents
        documents = pd.DataFrame({"Document": docs,
                                "ID": range(len(docs)),
                                "Topic": topics})

        documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
        cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)

        # Extract vectorizer and analyzer from BERTopic
        vectorizer = topic_model.vectorizer_model
        analyzer = vectorizer.build_analyzer()

        # Extract features for Topic Coherence evaluation
        words = vectorizer.get_feature_names_out()
        tokens = [analyzer(doc) for doc in cleaned_docs]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(token) for token in tokens]

        topic_words = [[words for words, _ in topic_model.get_topic(topic)] 
                     for topic in range(len(set(topics))-1)]

        coherence_model = CoherenceModel(topics=topic_words, texts=tokens, corpus=corpus,
                                       dictionary=dictionary, coherence='c_v')
        
        print(f"appending coherence {n_topics}")
        score = coherence_model.get_coherence()
        coherence_values.append(score)

        print(f"topic {n_topics} scores calculated, coherence score: {score}")

    return coherence_values

phases = ["phase 0", "phase 1", "phase 2", "phase 3", "phase 4", "phase 5", "phase 6"]

for phase in phases
    df_sub = df[df['phase'] == phase]
    coherence_values = compute_coherence_values(150, 35, 15, df_sub["processed_message"])

    fig, ax = plt.subplots(figsize=(8, 6))
    limit=150; start=35; step=15;
    x = range(start, limit, step)
    sns.set(color_codes=True)
    sns.lineplot(x= x, y=coherence_values)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    x_list = range(math.floor(min(x)), math.ceil(max(x))+1)
    plt.xticks(x);