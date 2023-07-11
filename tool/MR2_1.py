import random
import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet
nlp = spacy.load('en_core_web_sm')


class MR2_1(object):

    def generate(self, question, article):
        word = []
        place = []
        num = 0
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.text for token in tokens]
        for token in tokens:
            if token.tag_ in ["JJ", "JJR", "JJS"]:
                word.append(token.text)
                place.append(num)
            num += 1
        all_replaced_words = []
        for this_index in range(len(word)):
            if word[this_index] in ['same', 'legal']:
                continue
            synonyms = []
            for syn in wordnet.synsets(word[this_index], "a"):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().lower())
            if (len(synonyms) == 0):
                continue
            replaced_words = []
            for this_synonym in synonyms:
                if this_synonym == word[this_index]:
                    continue
                else:
                    if [word[this_index], this_synonym, place[this_index]] in replaced_words:
                        continue
                    replaced_words.append([word[this_index], this_synonym, place[this_index]])
            if replaced_words:
                all_replaced_words.append(replaced_words)
        if all_replaced_words == []:
            return question, article, False
        follow_question = real_tokens[:]
        for this_replaced_word_pair in all_replaced_words:
            num_of_synonym = len(this_replaced_word_pair)
            choose_index = random.randint(0,num_of_synonym-1)
            follow_question[this_replaced_word_pair[choose_index][2]] = this_replaced_word_pair[choose_index][1]
        this_follow_question = TreebankWordDetokenizer().detokenize(follow_question)
        if follow_question == real_tokens:
            return question, article, False
        return this_follow_question, article, True