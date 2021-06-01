import json
import spacy
import nltk
from nltk import Tree
from pattern.en import conjugate, lemma, lexeme, PRESENT, INFINITIVE, PAST, FUTURE, SG, PLURAL, PROGRESSIVE
nlp = spacy.load('en_core_web_sm')

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def name_find(node, list):
    if node.n_lefts + node.n_rights > 0:
        list.append(node.orth_)
        return [name_find(child, list) for child in node.children]
    else:
        list.append(node.orth_)


def add_not(sent_in_doc):
    for sent in sent_in_doc.sents:
        vb = str(sent.root)
        for token in sent:
            start = str(token)
            break
        vb_is_be = conjugate(vb, tense=INFINITIVE)
        if start =="have" and " ever " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" ever ", " have not ever ")
            return sent_not
        if start =="has" and " ever " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" ever ", " has not ever ")
            return sent_not
        if start in ["do", "did", "does"]:
            sent_0 = sent
            if start == "do":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "do not "+vb)
                return sent_not
            elif start == "did":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "did not " + vb)
                return sent_not
            elif start == "does":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "does not " + vb)
                return sent_not
        if start in ["have", "has"]:
            sent_0 = sent
            if start == "have":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "have not " + vb)
                return sent_not
            elif start == "has":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "has not " + vb)
                return sent_not
        if start in ["can", "could", "will", "would"]:
            sent_0 = sent
            if start == "can":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "can not " + vb)
                return sent_not
            elif start == "could":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "could not " + vb)
                return sent_not
            elif start == "will":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "will not " + vb)
                return sent_not
            elif start == "would":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "would not " + vb)
                return sent_not
        if start == "is" and " based " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" based ", " is not based ")
            return sent_not
        if start == "are" and " based " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" based ", " are not based ")
            return sent_not
        if start == "was" and " based " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" based ", " was not based ")
            return sent_not
        if start == "were" and " based " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" based ", " were not based ")
            return sent_not
        if start == "is" and " filmed " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" filmed ", " is not filmed ")
            return sent_not
        if start == "is" and " part of " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" part of ", " is not part of ")
            return sent_not
        if start == "is" and " the same " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" the same ", " is not the same ")
            return sent_not
        if start == "are" and " the same " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" the same ", " are not the same ")
            return sent_not
        if start == "is" and " still " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" still ", " is not still ")
            return sent_not
        if start == "are" and " still " in str(sent):
            sent_0 = sent
            sent_1 = sent_0[1:]
            sent_ken = str(sent_1)
            sent_not = sent_ken.replace(" still ", " are not still ")
            return sent_not

        if start in ["is", "are", "was", "were"] and vb_is_be != "be":
            sent_0 = sent
            if start == "is":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "is not " + vb)
                return sent_not
            elif start == "are":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "are not " + vb)
                return sent_not
            elif start == "was":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "was not " + vb)
                return sent_not
            elif start == "were":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                sent_not = sent_ken.replace(vb, "were not " + vb)
                return sent_not
        if start in ["is", "are", "was", "were"] and vb_is_be == "be":
            sent_0 = sent
            if start == "is":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                for child in sent.root.children:
                    name = child
                    break
                name_list = []
                name_find(name, name_list)
                sent_1 = sent_0[1:]
                part_1 = sent_1[0:len(name_list)]
                part_2 = sent_1[len(name_list):]
                sent_not = str(part_1) + " is not " + str(part_2)
                return sent_not
            elif start == "are":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                for child in sent.root.children:
                    name = child
                    break
                name_list = []
                name_find(name, name_list)
                sent_1 = sent_0[1:]
                part_1 = sent_1[0:len(name_list)]
                part_2 = sent_1[len(name_list):]
                sent_not = str(part_1) + " are not " + str(part_2)
                return sent_not
            elif start == "was":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                for child in sent.root.children:
                    name = child
                    break
                name_list = []
                name_find(name, name_list)
                sent_1 = sent_0[1:]
                part_1 = sent_1[0:len(name_list)]
                part_2 = sent_1[len(name_list):]
                sent_not = str(part_1) + " was not " + str(part_2)
                return sent_not
            elif start == "were":
                sent_1 = sent_0[1:]
                sent_ken = str(sent_1)
                for child in sent.root.children:
                    name = child
                    break
                name_list = []
                name_find(name, name_list)
                sent_1 = sent_0[1:]
                part_1 = sent_1[0:len(name_list)]
                part_2 = sent_1[len(name_list):]
                sent_not = str(part_1) + " were not " + str(part_2)
                return sent_not


def really_main(doc_in, sent_out):
    doc = nlp(doc_in)
    for sent in doc.sents:
        sent = nlp(str(sent))
        sent_out = add_not(sent)
        return sent_out


class MR1_4(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        output = []
        line = data
        this_line = json.loads(line.strip("\n"))
        question = this_line["question"]
        sent_out = ""
        sent_out = really_main(question, sent_out)
        if "all " in question:
            return
        elif " all " in question:
            return
        elif "every " in question:
            return
        elif " every " in question:
            return
        else:
            out_line = line.replace("\"question\": \"" + str(question),
                                    "\"question\": \"" + str(sent_out) + ", is it right", 1)
            output.append([line.strip(), out_line.strip()])
        return output

