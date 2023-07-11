import spacy
import random
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pattern.text import PAST, PROGRESSIVE, conjugate
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
        if child_tag in ["NN", "NNP"]:
            singular = True
        elif child_tag == "PRP":
            if str(child) in ["him", "her", "it", "me"]:
                singular = True
            else:
                singular = False
        elif child_tag in ["NNS", "NNPS"]:
            singular = False
    return singular


class MR1_2(object):

    def generate(self, question, article):
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.text for token in tokens]
        legal_first_token = ["did", "have", "has", "had", "will", "would"]
        first_token = str(tokens[0].text).strip()
        follow_question = real_tokens[:]
        vb_wrod = ""
        vb_wrod_lemma = ""
        if first_token in legal_first_token:
            is_single = True
            for word in doc:
                child_dep = [child.dep_ for child in list(word.children)]
                child_tag = [child.tag_ for child in list(word.children)]
                place_tree1 = 0
                if word.dep_ == "ROOT":
                    vb_wrod = str(word.text)
                    if word.lemma_ == 'been':
                        vb_wrod_lemma = 'be'
                    else:
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
            vb_wrod_index = 0
            for index in range(len(follow_question)):
                if follow_question[index] == vb_wrod:
                    vb_wrod_index = index
                    break
            if first_token in ["did"]:
                follow_question = real_tokens
                follow_question[0] = 'will'
            elif first_token in ["has", "had", "have"] and vb_wrod_index == 0:
                vb_wrod_index = -1
                pos = [token.pos_ for token in doc]
                for index in range(len(pos)):
                    if index in [0, 1]:
                        continue
                    if pos[index] in ['VERB', 'AUX']:
                        vb_wrod_index = index
                        break
                if vb_wrod_index == -1:
                    return question, article, False
                if tokens[vb_wrod_index].lemma_ == 'been':
                    vb_wrod_lemma = 'be'
                else:
                    vb_wrod_lemma = str(tokens[vb_wrod_index].lemma_)
                which = random.randint(0, 1)
                if which:
                    follow_question[0] = "will"
                    follow_question0 = follow_question[:vb_wrod_index]
                    follow_question1 = [vb_wrod_lemma]
                    if vb_wrod_index != len(follow_question) - 1:
                        follow_question1.extend(follow_question[vb_wrod_index + 1:])
                    follow_question0.extend(follow_question1)
                    follow_question = follow_question0
                    if 'ever' in follow_question:
                        follow_question.remove('ever')
                else:
                    if is_single:
                        follow_question[0] = "is"
                        follow_question0 = follow_question[:vb_wrod_index]
                        follow_question1 = ['going', 'to', vb_wrod_lemma]
                        if vb_wrod_index != len(follow_question) - 1:
                            follow_question1.extend(follow_question[vb_wrod_index + 1:])
                        follow_question0.extend(follow_question1)
                        follow_question = follow_question0
                        if 'ever' in follow_question:
                            follow_question.remove('ever')
                    else:
                        follow_question[0] = "are"
                        follow_question0 = follow_question[:vb_wrod_index]
                        follow_question1 = ['going', 'to', vb_wrod_lemma]
                        if vb_wrod_index != len(follow_question) - 1:
                            follow_question1.extend(follow_question[vb_wrod_index + 1:])
                        follow_question0.extend(follow_question1)
                        follow_question = follow_question0
                        if 'ever' in follow_question:
                            follow_question.remove('ever')
            elif first_token in ["has", "had", "have"]:
                vb_wrod_index = 0
                for index in range(len(follow_question)):
                    if follow_question[index] == vb_wrod:
                        vb_wrod_index = index
                        break
                if vb_wrod_index in [0, 1]:
                    vb_wrod_index = -1
                    pos = [token.pos_ for token in doc]
                    for index in range(len(pos)):
                        if index in [0, 1]:
                            continue
                        if pos[index] in ['VERB', 'AUX']:
                            vb_wrod_index = index
                            vb_wrod_lemma = tokens[index].lemma_
                            break
                    if vb_wrod_index == -1:
                        return question, article, False
                which = random.randint(0, 1)
                if which:
                    follow_question[0] = "will"
                    follow_question0 = follow_question[:vb_wrod_index]
                    follow_question1 = [vb_wrod_lemma]
                    if vb_wrod_index != len(follow_question) - 1:
                        follow_question1.extend(follow_question[vb_wrod_index + 1:])
                    follow_question0.extend(follow_question1)
                    follow_question = follow_question0
                    if 'ever' in follow_question:
                        follow_question.remove('ever')
                else:
                    if is_single:
                        follow_question[0] = "is"
                        follow_question0 = follow_question[:vb_wrod_index]
                        follow_question1 = ['going', 'to', vb_wrod_lemma]
                        if vb_wrod_index != len(follow_question)-1:
                            follow_question1.extend(follow_question[vb_wrod_index+1:])
                        follow_question0.extend(follow_question1)
                        follow_question = follow_question0
                        if 'ever' in follow_question:
                            follow_question.remove('ever')
                    else:
                        follow_question[0] = "are"
                        follow_question0 = follow_question[:vb_wrod_index]
                        follow_question1 = ['going', 'to', vb_wrod_lemma]
                        if vb_wrod_index != len(follow_question)-1:
                            follow_question1.extend(follow_question[vb_wrod_index+1:])
                        follow_question0.extend(follow_question1)
                        follow_question = follow_question0
                        if 'ever' in follow_question:
                            follow_question.remove('ever')
            elif first_token in ["will", "would"]:
                vb_PROGRESSIVE = conjugate(vb_wrod_lemma, tense=PAST, aspect=PROGRESSIVE)

                vb_wrod_index = 0
                for index in range(len(follow_question)):
                    if follow_question[index] == vb_wrod:
                        vb_wrod_index = index
                        break
                if vb_wrod_index in [0, 1]:
                    vb_wrod_index = -1
                    pos = [token.pos_ for token in doc]
                    for index in range(len(pos)):
                        if index in [0, 1]:
                            continue
                        if pos[index] in ['VERB', 'AUX']:
                            vb_wrod_index = index
                            vb_wrod_lemma = tokens[index].lemma_
                            vb_PROGRESSIVE = conjugate(vb_wrod_lemma, tense=PAST, aspect=PROGRESSIVE)
                            break
                    if vb_wrod_index == -1:
                        return question, article, False
                if is_single:
                    follow_question[0] = "has"
                    follow_question0 = follow_question[:vb_wrod_index]
                    follow_question1 = ['ever', vb_PROGRESSIVE]
                    if vb_wrod_index != len(follow_question)-1:
                        follow_question1.extend(follow_question[vb_wrod_index+1:])
                    follow_question0.extend(follow_question1)
                    follow_question = follow_question0
                else:
                    follow_question[0] = "have"
                    follow_question0 = follow_question[:vb_wrod_index]
                    follow_question1 = ['ever', vb_PROGRESSIVE]
                    if vb_wrod_index != len(follow_question)-1:
                        follow_question1.extend(follow_question[vb_wrod_index+1:])
                    follow_question0.extend(follow_question1)
                    follow_question = follow_question0
            if follow_question == real_tokens:
                return question, article, False
            else:
                this_follow_question = TreebankWordDetokenizer().detokenize(follow_question)
                return this_follow_question, article, True
        else:
            return question, article, False