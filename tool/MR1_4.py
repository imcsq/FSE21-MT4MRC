import spacy
from nltk import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer
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


def add_not(sent):
        tokens = [token for token in sent]
        real_tokens = [token.text for token in tokens]
        for word in sent:
            if word.dep_ == "ROOT":
                vb_doc = word
                vb = word.text
                vb_is_be = word.lemma_
        start = real_tokens[0]
        if start in ['has', 'have'] and 'ever' in real_tokens:
            ever_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'ever':
                    ever_index = index
                    break
            follow_question = real_tokens[1:ever_index]
            follow_question.extend([start, 'not', 'ever'])
            if ever_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[ever_index+1:])
            return follow_question, False
        if start in ['is', 'are', 'was', 'were'] and 'based' in real_tokens:
            vb_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'based':
                    vb_index = index
                    break
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', 'based'])
            if vb_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[vb_index+1:])
            return follow_question, False
        if start in ['is', 'are', 'was', 'were'] and 'filmed' in real_tokens:
            vb_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'filmed':
                    vb_index = index
                    break
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', 'filmed'])
            if vb_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[vb_index+1:])
            return follow_question, False
        if start in ['is', 'are', 'was', 'were'] and ' part of ' in TreebankWordDetokenizer().detokenize(real_tokens):
            vb_index = 0
            for index in range(len(real_tokens)-1):
                if real_tokens[index] == 'part' and real_tokens[index+1] == 'of':
                    vb_index = index
                    break
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', 'part', 'of'])
            if vb_index != len(real_tokens) - 2:
                follow_question.extend(real_tokens[vb_index+2:])
            return follow_question, False
        if start in ['is', 'are', 'was', 'were'] and " the same" in TreebankWordDetokenizer().detokenize(real_tokens):
            vb_index = 0
            for index in range(len(real_tokens)-1):
                if real_tokens[index] == 'the' and real_tokens[index+1] == 'same':
                    vb_index = index
                    break
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', 'the', 'same'])
            if vb_index != len(real_tokens) - 2:
                follow_question.extend(real_tokens[vb_index+2:])
            return follow_question, False
        if start in ['is', 'are', 'was', 'were'] and "still" in real_tokens:
            vb_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'still':
                    vb_index = index
                    break
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', 'still'])
            if vb_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[vb_index+1:])
            return follow_question, False
        if start in ["is", "are", "was", "were"] and vb_is_be != "be":
            vb_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == vb:
                    vb_index = index
                    break
            if vb_index in [0, 1]:
                return real_tokens, False
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', vb])
            if vb_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[vb_index+1:])
            return follow_question, False
        if start in ["is", "are", "was", "were"] and vb_is_be == "be":
            len_child = 0
            name = tokens[0]
            for child in vb_doc.children:
                len_child += 1
            if len_child == 1:
                return real_tokens, False
            for child in vb_doc.children:
                name = child
                break
            if name == tokens[0]:
                return real_tokens, False
            name_list = []
            name_find(name, name_list)
            part_1 = real_tokens[1:len(name_list)+1]  ######
            part_2 = real_tokens[len(name_list)+1:]
            follow_question = part_1
            follow_question.extend([start, 'not'])
            follow_question.extend(part_2)
            return follow_question, False
        if start in ["do", "did", "does", "have", "has", "can", "could", "will", "would", 'must', 'might', 'should']:
            vb_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == vb:
                    vb_index = index
                    break
            if vb_index in [0, 1]:
                vb_index = -1
                pos = [token.pos_ for token in tokens]
                for index in range(len(pos)):
                    if index in [0, 1]:
                        continue
                    if pos[index] in ['VERB', 'AUX']:
                        vb_index = index
                if vb_index == -1:
                    return real_tokens, False
            if real_tokens[vb_index-1] == 'be':
                vb_index = vb_index - 1
                vb = 'be'
                follow_question = real_tokens[1:vb_index]
                follow_question.extend([start, 'not', vb])
                if vb_index != len(real_tokens) - 1:
                    follow_question.extend(real_tokens[vb_index + 1:])
                return follow_question, False
            follow_question = real_tokens[1:vb_index]
            follow_question.extend([start, 'not', vb])
            if vb_index != len(real_tokens) - 1:
                follow_question.extend(real_tokens[vb_index + 1:])
            return follow_question, False


class MR1_4(object):

    def generate(self, question, article):
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.text for token in tokens]
        this_follow_question, if_out = add_not(doc)
        if "all" in real_tokens:
            return question, article, False
        elif "every" in real_tokens:
            return question, article, False
        if this_follow_question == real_tokens:
            return question, article, False
        else:
            this_follow_question.extend([',', 'is', 'it', 'right'])
            follow_question = TreebankWordDetokenizer().detokenize(this_follow_question)
            return follow_question, article, True