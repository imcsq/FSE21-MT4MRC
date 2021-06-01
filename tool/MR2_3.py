import json
import string
import nltk
import spacy
from nltk import Tree
from nltk.tokenize import word_tokenize
from pattern.en import conjugate, lemma, lexeme, PRESENT, INFINITIVE, PAST, FUTURE, SG, PLURAL, PROGRESSIVE
nlp = spacy.load('en_core_web_sm')


def first_letter(sent_in):
    if sent_in[0:2] == "I " or sent_in[0:2] == "I'":
        return sent_in
    else:
        sentence = word_tokenize(sent_in)
        sent = nltk.pos_tag(sentence)
        if sent[0][1] in ["NNP", "NNPS"]:
            return sent_in
        else:
            sent_in = sent_in[0].lower() + sent_in[1:]
            return sent_in


def new_first_letter(sent_in):
    if sent_in[0].isupper():
        return sent_in
    else:
        sent_in = sent_in[0].upper() + sent_in[1:]
        return sent_in


def punc_num(a_str):
    punc = string.punctuation
    num = 0
    for i in a_str:
        if i in punc and i != ",":
            num = num + 1
        elif i == ",":
            place = a_str.find(",")
            if a_str[place - 1] == " " and a_str[place + 1] == " ":
                num = num + 1
    return num


def special_punc_num(a_str):
    num = 0
    num = num + a_str.count("'s ")
    num = num + a_str.count("&")
    num = num + a_str.count(":\"")
    num = num + a_str.count(".")
    punc = string.punctuation
    if a_str[0] in punc:
        num = num + 1
    return num


def ex_prp(word):
    if word == 'I':
        return 'me'
    elif word == 'me':
        return 'I'
    if word == 'we':
        return 'us'
    elif word == 'us':
        return 'we'
    elif word == 'he':
        return 'him'
    elif word == 'him':
        return 'he'
    elif word == 'she':
        return 'her'
    elif word == 'her':
        return 'she'
    elif word == 'they':
        return 'them'
    elif word == 'them':
        return 'they'


def list_to_str(list_in):
    str_out = ""
    list_in = [str(i) for i in list_in]
    for i in list_in:
        str_out = str_out + i + " "
    return str_out


def tok_format(tok):
    return "_".join([tok.orth_, tok.dep_])


def tree_to_list(node, tree):
    if node.n_lefts + node.n_rights > 0:
        tree.insert(0, node.orth_)
        return [tree_to_list(child, tree) for child in node.children]
    else:
        tree.insert(0, node.orth_)


def ex_two_parts(sent_in, root_place_in_str, part1, part2, add_by):
    part1_used = ""
    part2_used = ""
    change1 = False
    change2 = False
    over_1 = False
    over_2 = False
    punc = string.punctuation
    part1_2 = part1[:]
    part2_2 = part2[:]
    if len(part1) == 1:
        over_1 = True
        part1_used = part1[0]
    if len(part2) == 1:
        over_2 = True
        part2_used = part2[0]
    for word in sent_in:
        if str(word) in part1 and change1 is False and over_1 is False:
            if str(word) in punc:
                part1_used = part1_used.rstrip() + str(word) + " "
            else:
                part1_used = part1_used + str(word) + " "
            this_place = part1.index(str(word))
            del part1[this_place]
            change1 = True
            continue
        if str(word) in part1 and change1 is True:
            if str(word) in punc:
                part1_used = part1_used.rstrip() + str(word) + " "
            else:
                part1_used = part1_used + str(word) + " "
            this_place = part1.index(str(word))
            del part1[this_place]
        elif str(word) not in part1 and change1 is True:
            punc_nums = punc_num(list_to_str(part1_2))
            special_punc_nums = special_punc_num(part1_used)
            if len(part1_used) != len(list_to_str(part1_2)) - punc_nums + special_punc_nums:
                change1 = False
                part1 = part1_2[:]
                part1_used = ""
            else:
                change1 = False
                over_1 = True
        if str(word) in part2 and change2 is False and over_2 is False:
            if str(word) in punc:
                part2_used = part2_used.rstrip() + str(word) + " "
            else:
                part2_used = part2_used + str(word) + " "
            this_place = part2.index(str(word))
            del part2[this_place]
            change2 = True
            continue
        if str(word) in part2 and change2 is True:
            if str(word) in punc:
                part2_used = part2_used.rstrip() + str(word) + " "
            else:
                part2_used = part2_used + str(word) + " "
            this_place = part2.index(str(word))
            del part2[this_place]
        elif str(word) not in part2 and change2 is True:
            punc_nums = punc_num(list_to_str(part2_2))
            special_punc_nums = special_punc_num(part2_used)
            if len(part2_used) != len(list_to_str(part2_2)) - punc_nums + special_punc_nums:
                change2 = False
                part2 = part2_2[:]
                part2_used = ""
            else:
                change2 = False
                over_2 = True
    if part1_used =="" or part2_used == "":
        return
    final_part1 = part1_used.split(" ")
    final_part2 = part2_used.split(" ")
    if final_part1[0] in ["I", "he", "she", "they", "we"]:
        final_part1[0] = ex_prp(final_part1[0])
    if final_part2[0] in ["me", "him", "them", "us"]:
        final_part2[0] = ex_prp(final_part2[0])
    part1_used = part1_used.rstrip()
    part2_used = part2_used.rstrip()
    part1_used = part1_used.replace(" 's ", "'s ")
    part2_used = part2_used.replace(" 's ", "'s ")
    part1_used = part1_used.replace("( ", " (")
    part2_used = part2_used.replace("( ", " (")
    part1_used = part1_used.replace("- ", "-")
    part2_used = part2_used.replace("- ", "-")
    part1_used = part1_used.replace("$ ", " $")
    part2_used = part2_used.replace("$ ", " $")
    part1_used = part1_used.replace("\" ", " \"", 1)
    part2_used = part2_used.replace("\" ", " \"", 1)
    punc = string.punctuation
    if part1_used[0] == " " and part1_used[1] in punc:
        part1_used = part1_used[1:]
    if part2_used[0] == " " and part2_used[1] in punc:
        part2_used = part2_used[1:]
    if part1_used[0] == " ":
        part1_used = part1_used[1:]
    if part2_used[0] == " ":
        part2_used = part2_used[1:]
    sent_in = str(sent_in)
    maybe_partplace_first = sent_in.find(part1_used)
    maybe_partpace_second = sent_in.find(part1_used, maybe_partplace_first + 1)
    maybe_partpace_third = sent_in.find(part1_used, maybe_partpace_second + 1)
    if maybe_partpace_second == -1:
        maybe_partpace_second = -1
        maybe_partpace_third = -1
    choice = 1
    if maybe_partpace_second == maybe_partplace_first:
        choice = 1
    else:
        if root_place_in_str - maybe_partpace_second < 0 or maybe_partpace_second == -1:
            choice = 1
        else:
            choice = 2
    if maybe_partpace_third != -1 and root_place_in_str - maybe_partpace_third > 0:
        choice = 3
    sent_in = sent_in.replace(part1_used, "@#$%", choice)
    if choice == 2:
        sent_in = sent_in.replace("@#$%", part1_used, 1)
    if choice == 3:
        sent_in = sent_in.replace("@#$%", part1_used, 2)
    root_place_in_str = root_place_in_str - len(part1_used) + len("@#$%")
    maybe_partplace_first = sent_in.find(part2_used)
    maybe_partpace_second = sent_in.find(part2_used, maybe_partplace_first + 1)
    maybe_partpace_third = sent_in.find(part1_used, maybe_partpace_second + 1)
    if maybe_partpace_second == -1:
        maybe_partpace_second = -1
        maybe_partpace_third = -1
    choice = 1
    if maybe_partpace_second == maybe_partplace_first:
        choice = 1
    else:
        if root_place_in_str - maybe_partplace_first > 0:
            choice = 2
        else:
            choice = 1
    if maybe_partpace_third != -1 and root_place_in_str - maybe_partpace_second > 0:
        choice = 3
    sent_in = sent_in.replace(part2_used, "%$#@", choice)
    if choice == 2:
        sent_in = sent_in.replace("%$#@", part1_used, 1)
    if choice == 3:
        sent_in = sent_in.replace("%$#@", part1_used, 2)
    part_final1 = ""
    part_final2 = ""
    for i in final_part1:
        part_final1 = part_final1 + i + " "
    for i in final_part2:
        part_final2 = part_final2 + i + " "
    part1_used = part_final1.rstrip()
    part2_used = part_final2.rstrip()
    part1_used = part1_used.replace(" 's ", "'s ")
    part2_used = part2_used.replace(" 's ", "'s ")
    part1_used = part1_used.replace("( ", " (")
    part2_used = part2_used.replace("( ", " (")
    part1_used = part1_used.replace("- ", "-")
    part2_used = part2_used.replace("- ", "-")
    part1_used = part1_used.replace("$ ", " $")
    part2_used = part2_used.replace("$ ", " $")
    part1_used = part1_used.replace("\" ", " \"", 1)
    part2_used = part2_used.replace("\" ", " \"", 1)
    if part1_used[0] == " ":
        part1_used = part1_used[1:]
    if part2_used[0] == " ":
        part2_used = part2_used[1:]
    sent_in = sent_in.replace("@#$%", part2_used)
    if add_by:
        if part1_used == "who":
            part1_used = "whom"
        sent_in = sent_in.replace("%$#@", "by" + " " + part1_used)
    else:
        sent_in = sent_in.replace("%$#@", part1_used)
    return sent_in


def find_word_in_sent(sent_in, place_in):
    i = -1
    for word_out in sent_in:
        i = i+1
        if i == place_in:
            return word_out


def singular_and_now(sent_in, place_real2, word_dep, is_by):
    singular = True
    now = True
    is_i = False
    if is_by:
        tree = []
        tree_to_list(find_word_in_sent(sent_in, place_real2), tree)
        if "and" in tree:
            singular = False
        if singular:
            for child in find_word_in_sent(sent_in, place_real2).children:
                if child.tag_ == "NN":
                    singular = True
                elif child.tag_ == "PRP":
                    if str(child) in ["him", "her", "it"]:
                        singular = True
                    elif str(child) in ["me"]:
                        singular = True
                        is_i = True
                    else:
                        singular = False
                elif child.tag_ == "NNS":
                    singular = False
    else:
        tree = []
        tree_to_list(find_word_in_sent(sent_in, place_real2), tree)
        if "and" in tree:
            singular = False
        if singular:
            child = find_word_in_sent(sent_in, place_real2)
            if child.tag_ == "NN":
                singular = True
            elif child.tag_ == "PRP":
                if str(child) in ["him", "her", "it"]:
                    singular = True
                elif str(child) in ["me"]:
                    singular = True
                    is_i = True
                else:
                    singular = False
            elif child.tag_ == "NNS":
                singular = False
    sent_dep = [token.dep_ for token in sent_in]
    place_3 = sent_dep.index(word_dep)
    sent_tag = [token.tag_ for token in sent_in]
    if sent_tag[place_3] == "VBD" or sent_tag[place_3] == "VBN":
        now = False
    return singular, now, is_i


def ex_vb(vb_word, now, singular, is_i, want_now, want_vb, want_vbn):
    if want_now:
        return conjugate(vb_word, tense=PRESENT, aspect=PROGRESSIVE)
    if want_vb:
        return conjugate(vb_word, tense=INFINITIVE)
    if want_vbn:
        return conjugate(vb_word, tense=PAST, aspect=PROGRESSIVE)
    if now:
        if is_i:
            if singular:
                return conjugate(vb_word, tense=PRESENT, person=1, number=SG)
            else:
                return conjugate(vb_word, tense=PRESENT, person=1, number=PLURAL)
        else:
            if singular:
                return conjugate(vb_word, tense=PRESENT, number=SG)
            else:
                return conjugate(vb_word, tense=PRESENT, number=PLURAL)
    else:
        if is_i:
            return conjugate(vb_word, person=1, tense=PAST)
        elif singular:
            return conjugate(vb_word, tense=PAST, number = SG)
        else:
            return conjugate(vb_word, tense=PAST, number = PLURAL)


def bz_be_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "auxpass", True)
    if is_neg[0]:
        sent_in = str(sent_in).replace(str(list(word.children)[place_real1]), "*&^%", 1) 
        sent_in = str(sent_in).replace("*&^%", ex_vb("do", now, singular, is_i, False, False, False), 1) 
    else:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " " + str(word) + " ", " " + ex_vb(str(word), now, singular, is_i, False, False, False) + " ", 1)  # 删除be
    sent_in = sent_in.replace(" " + str(list(word.children)[place_2]), "", 1) 
    sent_in = nlp(sent_in)
    del (tree2[-1])
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, False)


def bz_can_be_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    child_is_i = [0]
    singular, now, is_i = singular_and_now(sent_in, place_real2, "auxpass", True)
    sent_in = str(sent_in).replace(" " + str(word) + " ", " " + ex_vb(str(word), now, singular, is_i, False, True, False) + " ", 1)
    sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " ", 1) 
    sent_in = sent_in.replace(" " + str(list(word.children)[place_2]), "", 1)
    sent_in = nlp(sent_in)
    del (tree2[-1])
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, False)


def bz_can_have_been_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " ", 1) 
    sent_in = sent_in.replace(" " + str(list(word.children)[place_2]), "", 1)
    sent_in = nlp(sent_in)
    del (tree2[-1])
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, False)


def bz_be_being_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "aux", True)
    sent_in = str(sent_in).replace(" " + str(word) + " ", " " + ex_vb(str(word), now, singular, is_i, True, False, False) + " ", 1)
    if is_neg[0]:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1 - 2]),
                                       " " + ex_vb(str(list(word.children)[place_real1 - 2]), now, singular, is_i,
                                                   False, False, False), 1)
    else:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1 - 1]),
                                       " " + ex_vb(str(list(word.children)[place_real1 - 1]), now, singular, is_i,
                                                   False, False, False), 1) 
    if is_neg[0]:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " ", 1)
    else:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " ", 1) 
    sent_in = sent_in.replace(" " + str(list(word.children)[place_2]), "", 1)
    sent_in = nlp(sent_in)
    del (tree2[-1])
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, False)


def bz_have_been_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "aux", True)
    sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1 - 1]),
                                   " " + ex_vb("have", now, singular, is_i, False, False, False), 1)
    sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " ", 1)
    sent_in = sent_in.replace(" " + str(list(word.children)[place_2]) + " ", " ", 1)
    sent_in = nlp(sent_in)
    del (tree2[-1])
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, False)


def zb_do(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "ROOT", False)
    new_word = ex_vb(str(word), now, singular, is_i, False, False, True)
    sent_in = str(sent_in).replace(" " + str(word), " " + new_word, 1)
    if is_neg[0]:
        sent_in = str(sent_in).replace(str(list(word.children)[palce_1 + 1]), "*&^%", 1) 
        sent_in = str(sent_in).replace("*&^%", ex_vb("be", now, singular, is_i, False, False, False), 1) 
    else:
        sent_in = str(sent_in).replace(" " + new_word + " ", " " + ex_vb("be", now, singular, is_i, False, False, False) + " " + new_word + " ", 1)
    sent_in = nlp(sent_in)
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, True)


def zb_can_do(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word)) 
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "ROOT", False)
    new_word = ex_vb(str(word), now, singular, is_i, False, False, True)
    sent_in = str(sent_in).replace(" " + str(word), " " + new_word, 1)
    if is_neg[0]:
        sent_in = str(sent_in).replace(str(list(word.children)[place_real1]) + " ", str(list(word.children)[place_real1]) + " " + "be" + " ", 1)
    else:
        sent_in = str(sent_in).replace(" " + str(list(word.children)[place_real1]) + " ", " " + str(list(word.children)[place_real1]) + " " + "be" + " ", 1)
    sent_in = nlp(sent_in)
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, True)


def zb_can_have_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    sent_in = str(sent_in).replace(" " + "have" + " ", " have been" + " ", 1)
    sent_in = nlp(sent_in)
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, True)


def zb_be_doing(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "ROOT", False)
    new_word = ex_vb(str(word), now, singular, is_i, False, False, True)
    sent_in = str(sent_in).replace(" " + str(word) + " ", " " + new_word + " ", 1)
    sent_in = str(sent_in).replace(" " + new_word + " ", " " + 'being' + " " + new_word + " ", 1)
    sent_in = str(sent_in).replace(str(list(word.children)[place_real1]), "*&^%", 1)  
    sent_in = str(sent_in).replace("*&^%", ex_vb("be", now, singular, is_i, False, False, False), 1)  
    sent_in = nlp(sent_in)
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, True)


def zb_have_done(sent_in, word, tree1, tree2, palce_1, place_2, place_real1, place_real2, is_neg):
    root_place_in_str = str(sent_in).find(str(word))
    tree_to_list(list(word.children)[palce_1], tree1)
    tree_to_list(list(word.children)[place_2], tree2)
    singular, now, is_i = singular_and_now(sent_in, place_real2, "aux", False)
    new_word = ex_vb("have", now, singular, is_i, False, False, False)
    sent_in = str(sent_in).replace(str(list(word.children)[place_real1]), new_word, 1)
    if is_neg[0]:
        sent_in = str(sent_in).replace(" " + str(word) + " ", " " + "been" + " " + str(word) + " ", 1)
    else:
        sent_in = str(sent_in).replace(" " + str(word) + " ", " " + "been" + " " + str(word) + " ", 1)
    sent_in = nlp(sent_in)
    return ex_two_parts(sent_in, root_place_in_str, tree1, tree2, True)


def bei_to_zhu(sent_in_doc):
    sent_dep = [token.dep_ for token in sent_in_doc]
    sent_tag = [token.tag_ for token in sent_in_doc]
    if str(sent_in_doc[0]) == "how":
        return
    if str(sent_in_doc[0]) == "why":
        return
    if sent_dep[0] == "dobj" and sent_dep[1] == "nsubj":
        return
    if sent_tag[0] in ["WP", "WP$"]:
        for i in sent_in_doc:
            if str(i) in ["do", "did", "does"]:
                return
    for word in sent_in_doc:
        child_dep = [child.dep_ for child in list(word.children)]
        child_tag = [child.tag_ for child in list(word.children)]
        if word.tag_ == "VBN" and word.dep_ == "ROOT" and "auxpass" in child_dep and "agent" in child_dep:
            if "nsubjpass" in child_dep:
                if child_dep.count("nsubjpass") > 1:
                    return
                place_tree1 = child_dep.index("nsubjpass")
            elif "csubjpass" in child_dep:
                if child_dep.count("csubjpass") > 1:
                    return
                place_tree1 = child_dep.index("csubjpass")
            else: return
            if "aux" in child_dep:
                place_of_aux = child_dep.index("aux")
                if place_of_aux < place_tree1:
                    return
            place_tree2 = child_dep.index("agent")
            if "nsubjpass" in sent_dep:
                place_really1 = sent_dep.index("nsubjpass")
            elif "csubjpass" in sent_dep:
                place_really1 = sent_dep.index("csubjpass")
            else: return
            place_really2 = sent_dep.index("agent")
            if place_tree1 > place_tree2:
                return
            tree1 = []
            tree2 = []
            is_neg = [0]
            if child_dep.count("aux") == 0:
                if child_dep.count("neg") != 0 and str(list(word.children)[0]) != "not":
                    is_neg[0] = 1
                place_really1 = child_dep.index("auxpass")
                sent_out = bz_be_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2, place_really1,
                                      place_really2, is_neg)
                return sent_out
            elif child_dep.count("aux") == 2:
                if child_dep.count("neg") != 0 and str(list(word.children)[0]) != "not":
                    is_neg[0] = 1
                place_really1 = child_dep.index("auxpass")
                sent_out = bz_can_have_been_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                                 place_really1, place_really2, is_neg)
                return sent_out
            elif child_dep.count("aux") == 1:
                if child_dep.count("neg") != 0 and str(list(word.children)[0]) != "not":
                    is_neg[0] = 1
                if 'MD' in child_tag:
                    place_really1 = child_dep.index("auxpass")
                    sent_out = bz_can_be_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                              place_really1, place_really2, is_neg)
                    return sent_out
                elif 'VBG' in child_tag:
                    place_really1 = child_dep.index("auxpass")
                    sent_out = bz_be_being_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                                place_really1, place_really2, is_neg)
                    return sent_out
                elif 'VBN' in child_tag:
                    place_really1 = child_dep.index("auxpass")
                    sent_out = bz_have_been_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                                 place_really1, place_really2, is_neg)
                    return sent_out


def zhu_to_bei(sent_in_doc):
    sent_dep = [token.dep_ for token in sent_in_doc]
    sent_tag = [token.tag_ for token in sent_in_doc]
    if str(sent_in_doc[0]) == "how":
        return
    if str(sent_in_doc[0]) == "why":
        return
    if sent_dep[0] == "dobj" and sent_dep[1] == "nsubj":
        return
    if sent_tag[0] in ["WP", "WP$"]:
        for i in sent_in_doc:
            if str(i) in ["do", "did", "does"]:
                return
    for word in sent_in_doc:
        child_dep = [child.dep_ for child in list(word.children)]
        child_tag = [child.tag_ for child in list(word.children)]
        if "VB" in word.tag_ and word.dep_ == "ROOT" and "auxpass" not in child_dep and "agent" not in child_dep and "nsubj" in child_dep and "dobj" in child_dep:  # [By, ,, I, groups, at, .]
            place_tree1 = child_dep.index("nsubj")
            if child_dep.count("nsubj") > 1:
                return
            if "aux" in child_dep:
                place_of_aux = child_dep.index("aux")
                if place_of_aux < place_tree1:
                    return
            place_tree2 = child_dep.index("dobj")
            if place_tree1 > place_tree2:
                return
            place_really1 = sent_dep.index("nsubj")
            place_really2 = sent_dep.index("dobj")
            tree1 = []
            tree2 = []
            is_neg = [0]
            if child_dep.count("aux") == 0: 
                sent_out = zb_do(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2, place_really1,
                                 place_really2, is_neg)
                return sent_out
            elif child_dep.count("aux") == 1 and child_dep.count(
                    "neg") != 0 and word.tag_ == "VB" and 'MD' not in child_tag and str(list(word.children)[0]) != "not":
                is_neg[0] = 1
                sent_out = zb_do(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2, place_really1,
                                 place_really2, is_neg)
                return sent_out
            elif child_dep.count("aux") == 2:
                if child_dep.count("neg") != 0 and str(list(word.children)[0]) != "not":
                    is_neg[0] = 1
                sent_out = zb_can_have_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                            place_really1, place_really2, is_neg)
                return sent_out
            elif child_dep.count("aux") == 1:
                if child_dep.count("neg") != 0 and str(list(word.children)[0]) != "not":
                    is_neg[0] = 1
                if 'MD' in child_tag:
                    if is_neg[0]:
                        place_really1 = child_dep.index("neg")
                    else:
                        place_really1 = child_dep.index("aux")
                    sent_out = zb_can_do(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                         place_really1, place_really2, is_neg)
                    return sent_out
                elif word.tag_ == 'VBG':
                    place_really1 = child_dep.index("aux")
                    sent_out = zb_be_doing(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                           place_really1, place_really2, is_neg)
                    return sent_out
                elif word.tag_ == 'VBN' or word.tag_ == 'VBD':
                    place_really1 = child_dep.index("aux")
                    sent_out = zb_have_done(sent_in_doc, word, tree1, tree2, place_tree1, place_tree2,
                                            place_really1, place_really2, is_neg)
                    return sent_out


def zhuangyu_ahead(sent_in_doc):
    sent_dep = [token.dep_ for token in sent_in_doc]
    sent_tag = [token.tag_ for token in sent_in_doc]
    place_of_root = sent_dep.index("ROOT")
    root_tag = sent_tag[place_of_root]
    if root_tag == "VBZ":
        return
    elif "VB" in root_tag:
        return


class MR2_3(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        output = []
        line = data
        this_line = json.loads(line.strip("\n"))
        pas = this_line["passage"]
        passage = pas
        passage_2 = pas
        sentc = nlp(pas)
        sents0 = []
        sents1 = []
        for sent in sentc.sents:
            se = nlp(str(sent))
            for sen in se.sents:
                sent0 = str(sen)[:]
                sen = first_letter(str(sen))
                sen = nlp(sen)
                sent0 = nlp(sent0)
                sent1 = bei_to_zhu(sen)
                if sent1 is not None:
                    sent1 = new_first_letter(sent1)
                    passage_2 = pas.replace(str(sent0), str(sent1))
                    sents0.append(sent0)
                    sents1.append(sent1)
                else:
                    sent1 = zhu_to_bei(sen)
                    if sent1 is None:
                        continue
                    else:
                        sent1 = new_first_letter(sent1)
                        passage_2 = pas.replace(str(sent0), str(sent1))
                        sents0.append(sent0)
                        sents1.append(sent1)
        if passage != passage_2:
            out_line = line.replace(str(passage), str(passage_2))
            if out_line != line:
                output.append([line.strip(), out_line.strip()])
        if output == []:
            return
        return output





