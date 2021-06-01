import json
import numpy
import spacy
import nltk
from nltk.corpus import wordnet
nlp = spacy.load('en_core_web_sm')

def tree_to_list(node, tree):
        if node.n_lefts + node.n_rights > 0:
            tree.insert(0, node.orth_)
            return [tree_to_list(child, tree) for child in node.children]
        else:
            tree.insert(0, node.orth_)

class MR1_1(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        output = []
        line = data
        tree1 = []
        this_line = json.loads(line)
        question = this_line["question"]
        doc = nlp(this_line["question"])
        tokens = [token for token in doc]
        real_tokens = [token.string.strip() for token in tokens]
        if str(tokens[0].string.strip()) not in ["is", "are", "was", "were"]:
            return
        if str(tokens[1].string.strip()) == "there":
            return
        if str(tokens[1].string.strip()) in ["any", "anyone", "anything"]:
            return
        sent_dep = [token.dep_ for token in doc]
        word_index = []
        for word in doc:
            child_dep = [child.dep_ for child in list(word.children)]
            child_tag = [child.tag_ for child in list(word.children)]
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
            return
        legal_pos = numpy.max(word_index)
        word = []
        place = []
        num = 0
        for token in tokens[legal_pos + 1:]:
            if token.tag_ in ["JJ", "JJR", "JJS"]:
                word.append(token.string.strip())
                place.append(legal_pos + 1 + num)
            num += 1
        rd = -1
        for one in word:
            rd += 1
            antonyms = []
            for syn in wordnet.synsets(one, "a"):
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            if (len(antonyms) == 0):
                continue
            else:
                if place[rd] == (len(tokens) - 1):
                    for ant in antonyms:
                        question_out = question
                        question_out = question_out.replace(" " + one, " " + ant, 1)
                        out_line2 = line
                        out_line2 = out_line2.replace(question, question_out, 1)
                        if line != out_line2:
                            if [line.strip(), out_line2.strip()] not in output:
                                output.append([line.strip(), out_line2.strip()])
                else:
                    for ant in antonyms:
                        question_out = question
                        question_out = question_out.replace(" " + one + " ", " " + ant + " ", 1)
                        out_line2 = line
                        out_line2 = out_line2.replace(question, question_out, 1)
                        if line != out_line2:
                            if [line.strip(), out_line2.strip()] not in output:
                                output.append([line.strip(), out_line2.strip()])
        if output == []:
            return
        return output
