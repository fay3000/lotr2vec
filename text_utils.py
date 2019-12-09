import re

STOPWORDS = {"but", "again", "there", "about", "an", "be", "for", "do", "its", "of", "while",
             "is", "s", "am", "or", "who", "as", "from", "the", "until", "are", "these", "were", "down",
             "should", "to", "had", "when", "at", "before", "and", "have", "in", "will", "on", "does",
             "then", "that", "because", "what", "why", "so", "can", "did", "has", "just", "where", "too",
             "which", "those", "i", "after", "whom", "t", "being", "if", "a", "by",
             "doing", "it", "how", "was", "here", "than", "don", "nor"}

# Converts text into array of words, removes
# punctuation & numbers, lowers case.
def word_tokenise(raw_text):
    clean = re.sub("[^a-zA-Z]", " ", raw_text).lower()
    words = clean.split()
    return words

def remove_stopwords(word_list):
    filtered_words = [w for w in word_list if
                      w not in STOPWORDS]
    return filtered_words
