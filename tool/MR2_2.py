import json
import numpy
import spacy
nlp = spacy.load('en_core_web_sm')

def tree_to_list(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)


class MR2_2(object):

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
        which = 0
        for token in tokens[legal_pos + 1:]:
            if token.dep_ in ["prep", "mark", "advmod"]:
                word.append(token.string.strip())
                which = real_tokens.index(token.string.strip())
        if word == []:
            return
        prep = ["when", "in", "at", "on", "if"]
        first_word = ["am", "is", "are", "was", "were"]
        if str(tokens[0].string).strip() in first_word and " there " not in question:
            return
        if word[-1] in prep:
            this_prep = " " + word[-1] + " "
            if this_prep not in question:
                return
            position = question.index(this_prep)
            part_1 = question[0:position]
            part_2 = question[position + 1:]
            sent = part_2 + ", " + part_1
            out_line = line.replace(question, sent, 1)
            if out_line != line:
                output.append([line.strip(), out_line.strip()])
        if output == []:
            return
        return output






