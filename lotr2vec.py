
# future lets us use python 2 and 3 features
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import codecs
import glob
import logging
import os
import gensim.models.word2vec as w2v
import nltk
import numpy as np
import multiprocessing

# Plotting
import plotly
import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage

import text_utils as utils

# Set up logging for model training info
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# # Download nltk tokeniser
# nltk.download("punkt")

# Load sentence tokeniser (uses nltk sentence boundary detection algorithm that learns abbrevs like 'dr.'.
# tokenizer._params.abbrev_types shows these abbreviations)
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

class Sentences(object):  # Memory-friendly iterator class.
    def __init__(self, dirname):
        self.source_filenames = sorted(glob.glob(dirname))

    def __iter__(self):
        # Avoids keeping list of all sentences in memory from all training data files.
        # We can read in each file, feed sentences to model, then forget them.
        # Note - model enters this loop multiple times because of epochs (won't increase word counts it considers):
        for source_filename in self.source_filenames:
            print("Reading '{0}'...".format(source_filename))

            with codecs.open(source_filename, "r", "utf-8") as file:  # use codecs to read in to utf8 format
                file_text = file.read()
                print("Raw corpus is {0} characters long".format(len(file_text)))
                # Split the text into sentences
                raw_sentences = tokenizer.tokenize(file_text)

                for raw_sentence in raw_sentences:
                    # Clean, tokenise then remove stopwords
                    sentence_words = utils.word_tokenise(raw_sentence)
                    sentence_words = utils.remove_stopwords(sentence_words)
                    yield sentence_words


sentences = Sentences("./data/lotr/lotr/*.txt")


print("Training lotr w2v Model...")

# Dimensionality of the resulting word vectors.
# 10-300 depending on training data size (recommended by Mikolov et al).
# More dims increases accuracy of word vectors but takes longer to train.
num_features = 300

# Minimum word count threshold
min_word_count = 3

# Number of threads to run in parallel
num_workers = multiprocessing.cpu_count()

# Context window length
context_size = 7

# Downsample setting for frequent words
downsampling = 1e-3

# Seed for the random number generator,
# to make the results reproducible
seed = 1

model = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

# Train model
model.build_vocab(sentences)
print("Word2Vec vocabulary length:", len(model.wv.vocab))
model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)

# Save model to file
if not os.path.exists("./trained"):
    os.makedirs("./trained")

model.save(os.path.join("trained", "lotr2vec.w2v"))

# Load model
model = w2v.Word2Vec.load(os.path.join("trained", "lotr2vec.w2v"))

print("Semantic Similarity Analysis")

# Find 10 most similar words to..
print("Most similar words to gandalf: {}".format(model.most_similar("gandalf")))
print("Most similar words to goblins: {}".format(model.most_similar('goblins')))
print("Most similar words to fire: {}".format(model.most_similar('fire')))
print("Most similar words to night: {}".format(model.most_similar('night')))

# Predict the 'odd one out'
gang = ['frodo', 'sam', 'pippin', 'sauron']
print("Odd one out from: {} is {}".format(gang, model.wv.doesnt_match(gang)))

# Predict word analogies using vector arithmetic.
# Mikolov et al famously found king - man + woman = queen
def nearest_similarity_cosmul(start1, end1, end2):
    # Calculate start1 - end1 + end2
    similarities = model.wv.most_similar_cosmul(
        positive=[end2, start1],
        negative=[end1]
    )
    start2 = similarities[0][0]
    print("{start1} is related to {end1}, as {start2} is related to {end2}".format(**locals()))
    return start2


print("Word analogies:")
nearest_similarity_cosmul('light', 'dark', 'evening')
# > 'light is to dark as morning is to evening'
nearest_similarity_cosmul('death', 'life', 'sky')
# > 'death is to life as cloud is to sky'
nearest_similarity_cosmul('king', 'man', 'woman')
# > inaccurate results: 'king is to man as gil is to woman'


# Create dendrogram of lotr characters
NAMES = {'Frodo',
        'Gandalf',
        'Sam',
        'Merry',
        'Pippin',
        'Aragorn',
        'Legolas',
        'Gimli',
        'Boromir',
        'Sauron',
        'Nazgul',
        'Gollum',
        'Bilbo',
        'Tom',
        'Glorfindel',
        'Elrond',
        'Arwen',
        'Galadriel',
        'Saruman',
        'Eomer',
        'Theoden',
        'Eowyn',
        'Wormtongue',
        'Shadowfax',
        'Treebeard',
        'Quickbeam',
        'Shelob',
        'Faramir',
        'Denethor',
        'Beregond',
        'Butterbur'}

def map_to_model_name(name):
    # names in the raw text and model
    # are encoded as the following
    name = name.lower()
    raw = {
        "nazgul": "nazgyl",
        "theoden": "thjoden",
        "eomer": "jomer",
        "eowyn": "jowyn",
    }
    # this sets default value to name
    return raw.get(name, name)


names = [map_to_model_name(name) for name in NAMES]
name_vectors = [model.wv[name] for name in names]
X = np.array(name_vectors)

# Note that create_dendrogram calls scipy.cluster.hierarchy.dendrogram
fig = ff.create_dendrogram(X,
                           orientation='left',
                           labels=list(NAMES),
                           color_threshold=0.3,
                           linkagefun=lambda x: linkage(X, 'complete', metric='cosine'))
fig['layout'].update({'width': 1000, 'height': 1000})
plotly.offline.plot(fig, filename='lotr_dendrogram.html')
