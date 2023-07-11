import numpy
import spacy
import random
from nltk.corpus import wordnet
from nltk.tokenize.treebank import TreebankWordDetokenizer
nlp = spacy.load('en_core_web_sm')


def tree_to_list(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)


class MR1_1(object):

    def generate(self, question, article):
        tree1 = []
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.text for token in tokens]
        if str(tokens[0].lemma_) != 'be':
            return question, article, False
        if str(tokens[1].text) == "there":
            return question, article, False
        if str(tokens[1].text) in ["any", "anyone", "anything"]:
            return question, article, False
        word_index = []
        for word in doc:
            child_dep = [child.dep_ for child in list(word.children)]
            if word.dep_ == "ROOT":
                if "nsubj" in child_dep:
                    place_tree1 = child_dep.index("nsubj")
                elif "nsubjpass" in child_dep:
                    place_tree1 = child_dep.index("nsubjpass")
                elif "csubj" in child_dep:
                    place_tree1 = child_dep.index("csubj")
                elif "csubjpass" in child_dep:
                    place_tree1 = child_dep.index("csubjpass")
                else:
                    continue
                tree_to_list(list(word.children)[place_tree1], tree1)
                for leaf in tree1:
                    word_index.append(real_tokens.index(leaf))
        if word_index == []:
            return question, article, False
        legal_pos = numpy.max(word_index)
        word = []
        place = []
        num = 0
        for token in tokens[legal_pos + 1:]:
            if token.tag_ in ["JJ", "JJR", "JJS"]:
                word.append(token.text)
                place.append(legal_pos + 1 + num)
            num += 1
        all_available_follow = []
        num = -1
        for this_word in word:
            num += 1
            antonyms = []
            for syn in wordnet.synsets(this_word, "a"):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            if not len(antonyms):
                continue
            else:
                for this_antonym in antonyms:
                    replaced_question = real_tokens[:]
                    replaced_question[place[num]] = this_antonym
                    all_available_follow.append(TreebankWordDetokenizer().detokenize(replaced_question))

        if all_available_follow == []:
            return question, article, False
        random.shuffle(all_available_follow)
        if all_available_follow[0] == real_tokens:
            return all_available_follow[0], article, False
        else:
            return all_available_follow[0], article, True