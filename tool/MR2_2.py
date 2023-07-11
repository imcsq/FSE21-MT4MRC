import benepar, spacy
from nltk import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer


def read_tree(tree):
    if isinstance(tree, str):
        return tree
    else:
        return [read_tree(i) for i in tree]


def read_list(a_list):
    one_dim_list = []
    for item in a_list:
        if isinstance(item, list):
            one_dim_list.extend(read_list(item))
        else:
            one_dim_list.append(item)
    return one_dim_list


nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

class MR2_2(object):

    def generate(self, question, article):
        doc = nlp(question)
        sentence = list(doc.sents)[0]
        t = Tree.fromstring(sentence._.parse_string)
        this_tree = read_tree(t)
        this_tree_list = []
        for i in this_tree:
            this_tree_list.append(read_list(i))
        if len(this_tree_list) == 1:
            return question, article, False
        for index in range(len(this_tree_list)):
            if this_tree_list[index][0] in ['when', 'while', 'since', 'because', 'if', 'in', "at"]:
                if index == 0:
                    final_sentence = []
                    for part in this_tree_list[1:]:
                        final_sentence.extend(part)
                    final_sentence.append(',')
                    final_sentence.extend(this_tree_list[index])
                    follow_question = TreebankWordDetokenizer().detokenize(final_sentence)
                    return follow_question, article, True
                if index == len(this_tree_list)-1:
                    final_sentence = []
                    final_sentence.extend(this_tree_list[index])
                    final_sentence.append(',')
                    for part in this_tree_list[:-1]:
                        final_sentence.extend(part)
                    follow_question = TreebankWordDetokenizer().detokenize(final_sentence)
                    return follow_question, article, True
        return question, article, False