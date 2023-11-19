import streamlit as st
from numpy.linalg import norm
import numpy as np
from sentence_transformers import SentenceTransformer

@st.cache_resource
def sentence_transformer(list_of_players, player):
        # Define the model we want to use (it'll download itself)
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = list_of_players

        # vector embeddings created from dataset
        embeddings = model.encode(sentences)

        # query vector embedding
        query_embedding = model.encode(player)

        # define our distance metric
        def cosine_similarity(a, b):
            return np.dot(a, b)/(norm(a)*norm(b))

        # run semantic similarity search
        for e, s in zip(embeddings, sentences):
            if cosine_similarity(e, query_embedding) > 0.8:
                result = s
                # print(s, " -> similarity score = ", cosine_similarity(e, query_embedding))

        return result






