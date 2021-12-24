from nltk.corpus import wordnet as wn
import pickle

with open('event_sense_mapping.pkl', "rb") as esm:
    pickle.load(event_sense_mapping_)