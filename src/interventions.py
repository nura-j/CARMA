import random
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
stops = set(stopwords.words('english'))

def get_synonyms(word, match_pos=True, match_tokenization=False, num_synonyms=5):
    synonyms = []
    for syn in wn.synsets(word):
        for lm in syn.lemmas():
            if len(synonyms) >= num_synonyms:
                break
            if lm.name() != word:
                if match_pos and lm.synset().pos() != syn.pos():
                    continue
                synonyms.append(lm.name())
    return synonyms


def synonym_replacement(sentence, n):
    sentence = sentence.split(' ')
    new_sentence = sentence.copy()
    random_word_list = list(set([word for word in sentence if word not in stops]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            # clean the synonym
            synonym = synonym.replace("_", " ")
            new_sentence = [synonym if word == random_word else word for word in new_sentence]
            num_replaced += 1
        if num_replaced >= n:
            break
    new_sentence = ' '.join(new_sentence)
    return new_sentence
