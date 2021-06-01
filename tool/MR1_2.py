import json
import spacy
import random
from pattern.en import conjugate, lemma, lexeme, PRESENT, INFINITIVE, PAST, FUTURE, SG, PLURAL, PROGRESSIVE
nlp = spacy.load('en_core_web_sm')


def tree_to_list(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)


def singular_and_now(tree, child, child_tag):
    singular = True
    if "and" in tree:
        singular = False
    if singular:
        if child_tag == "NN":
            singular = True
        elif child_tag == "PRP":
            if str(child) in ["him", "her", "it", "me"]:
                singular = True
            else:
                singular = False
        elif child_tag == "NNS":
            singular = False
    return singular


class MR1_2(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        output = []
        line = data
        this_line = json.loads(line.strip("\n"))
        word = []
        question = this_line["question"]
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.string.strip() for token in tokens]
        legal_first_token = ["did", "have", "has", "had", "will", "would"]
        first_token = str(tokens[0].string).strip()
        question_2 = ''
        vb_wrod = ""
        vb_wrod_lemma = ""
        if first_token in legal_first_token:
            is_single = True
            for word in doc:
                child_dep = [child.dep_ for child in list(word.children)]
                child_tag = [child.tag_ for child in list(word.children)]
                place_tree1 = 0
                if word.dep_ == "ROOT":
                    vb_wrod = str(word)
                    vb_wrod_lemma = str(word.lemma_)
                    if "nsubj" in child_dep:
                        place_tree1 = child_dep.index("nsubj")
                    elif "nsubjpass" in child_dep:
                        place_tree1 = child_dep.index("nsubjpass")
                    elif "csubj" in child_dep:
                        place_tree1 = child_dep.index("csubj")
                    elif "csubjpass" in child_dep:
                        place_tree1 = child_dep.index("csubjpass")
                if place_tree1:
                    tree1 = []
                    tree_to_list(list(word.children)[place_tree1], tree1)
                    is_single = singular_and_now(tree1, list(word.children)[place_tree1], child_tag[place_tree1])
            if first_token in ["did", "has", "had", "have"]:
                which = random.randint(0, 1)
                if which:
                    question_2 = question.replace(first_token + " ", "will ", 1)
                    question_2 = question_2.replace(" ever ", " ")
                else:
                    if is_single:
                        question_2 = question.replace(first_token + " ", "is ", 1)
                        question_2 = question_2.replace(" " + vb_wrod + " ", " going to " + vb_wrod_lemma + " ", 1)
                        question_2 = question_2.replace(" ever ", " ")
                    else:
                        question_2 = question.replace(first_token + " ", "are ", 1)
                        question_2 = question_2.replace(" " + vb_wrod + " ", " going to " + vb_wrod_lemma + " ", 1)
                        question_2 = question_2.replace(" ever ", " ")
            elif first_token in ["will", "would"]:
                vb_PROGRESSIVE = conjugate(vb_wrod_lemma, tense=PAST, aspect=PROGRESSIVE)
                if is_single:
                    question_2 = question.replace(first_token + " ", "has ", 1)
                    question_2 = question_2.replace(" " + vb_wrod + " ", " ever " + vb_PROGRESSIVE + " ", 1)

                else:
                    question_2 = question.replace(first_token + " ", "have ", 1)
                    question_2 = question_2.replace(" " + vb_wrod + " ", " ever " + vb_PROGRESSIVE + " ", 1)
            out_line = str(line).replace(question, question_2, 1)
            output.append([line.strip(), out_line.strip()])
            return output


