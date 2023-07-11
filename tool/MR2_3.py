import string
import spacy
from nltk import Tree
from nltk.tokenize.treebank import TreebankWordDetokenizer
from pattern.text.en import conjugate, PRESENT, INFINITIVE, PAST, SG, PLURAL, PROGRESSIVE

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})


def if_singular(tokens):
    tokens_text = [token.text for token in tokens]
    tokens_tag = [token.tag_ for token in tokens]
    singular = True
    is_me = False
    if "and" in tokens_text:
        singular = False
        return singular, is_me
    if ['NNS', 'NNPS'] in tokens_tag:
        return singular, is_me
    if tokens_text == ['me']:
        is_me = True
        return singular, is_me
    return singular, is_me


def if_now(a_token):
    if a_token.tag_ in ['VBD', 'VBN']:
        return False
    else:
        return True


def corresponding_tense_vb(vb_word, if_now, if_singular, if_me, want_now, want_vb, want_vbn):
    if want_now:
        return conjugate(vb_word, tense=PRESENT, aspect=PROGRESSIVE)
    if want_vb:
        return conjugate(vb_word, tense=INFINITIVE)
    if want_vbn:
        return conjugate(vb_word, tense=PAST, aspect=PROGRESSIVE)
    if if_now:
        if if_me:
            if if_singular:
                return conjugate(vb_word, tense=PRESENT, person=1, number=SG)
            else:
                return conjugate(vb_word, tense=PRESENT, person=1, number=PLURAL)
        else:
            if if_singular:
                return conjugate(vb_word, tense=PRESENT, number=SG)
            else:
                return conjugate(vb_word, tense=PRESENT, number=PLURAL)
    else:
        if if_me:
            return conjugate(vb_word, person=1, tense=PAST)
        elif if_singular:
            return conjugate(vb_word, tense=PAST, number=SG)
        else:
            return conjugate(vb_word, tense=PAST, number=PLURAL)


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


def ob_sb(word):
    if word == 'i':
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


class MR2_3(object):

    def generate(self, question, article):
        doc = nlp(article)
        new_article = []
        if_changed = False
        for sentence in doc.sents:
            tokens = [token for token in sentence]
            real_token = [token.text for token in sentence]
            t = Tree.fromstring(sentence._.parse_string)
            this_tree = read_tree(t)
            this_tree_list = []
            for i in this_tree:
                this_tree_list.append(read_list(i))
            sub_tree_label = [i.label() for i in t]
            if 'NP' not in sub_tree_label or 'VP' not in sub_tree_label:
                new_article.append(sentence.text)
                continue
            NP_index = sub_tree_label.index('NP')
            VP_index = sub_tree_label.index('VP')
            if VP_index < NP_index:
                new_article.append(sentence.text)
                continue
            split_by_punc = False
            for index in range(NP_index, VP_index):
                if sub_tree_label[index] in string.punctuation:
                    split_by_punc = True
            if split_by_punc:
                new_article.append(sentence.text)
                continue
            part_NP_indexes = []
            part_VP_indexes = []
            this_index = -1
            for index in range(len(this_tree_list)):
                if index not in [NP_index, VP_index]:
                    this_index += len(this_tree_list[index])
                elif index == NP_index:
                    part_NP_indexes.append(this_index + 1)
                    part_NP_indexes.append(this_index + len(this_tree_list[index]))
                    this_index += len(this_tree_list[index])
                elif index == VP_index:
                    part_VP_indexes.append(this_index + 1)
                    part_VP_indexes.append(this_index + len(this_tree_list[index]))
                    this_index += len(this_tree_list[index])
            if part_VP_indexes[1] - part_VP_indexes[0] <= 1:
                new_article.append(sentence.text)
                continue
            new_NP_if_now = if_now(tokens[part_VP_indexes[0]])
            if 'by' in real_token[part_VP_indexes[0]:part_VP_indexes[1] + 1]:
                if part_VP_indexes[1] - part_VP_indexes[0] < 3:
                    new_article.append(sentence.text)
                    continue
                num_by = 0
                for one in real_token[part_VP_indexes[0]:part_VP_indexes[1] + 1]:
                    if one == 'by':
                        num_by += 1
                if num_by != 1:
                    new_article.append(sentence.text)
                    continue
                part_VP_text = real_token[part_VP_indexes[0]:part_VP_indexes[1] + 1]
                part_VP_tag = [token.tag_ for token in tokens[part_VP_indexes[0]:part_VP_indexes[1] + 1]]
                if 'PRP' in part_VP_tag or 'PRP$' in part_VP_tag:
                    new_article.append(sentence.text)
                    continue
                by_index = part_VP_text.index('by') + part_VP_indexes[0]
                if 'VB' in tokens[by_index + 1].tag_:
                    new_article.append(sentence.text)
                    continue
                if tokens[part_VP_indexes[0]].lemma_ == 'be' and tokens[part_VP_indexes[0] + 1].tag_ == 'VBN':
                    if real_token[part_VP_indexes[0] + 2] == 'and':
                        new_article.append(sentence.text)
                        continue
                    new_NP_index = [by_index + 1, part_VP_indexes[1]]
                    new_NP_if_single, new_NP_if_me = if_singular(tokens[new_NP_index[0]: new_NP_index[1]])
                    new_vb = corresponding_tense_vb(tokens[part_VP_indexes[0] + 1].lemma_,
                                                    if_now=new_NP_if_now,
                                                    if_singular=new_NP_if_single,
                                                    if_me=new_NP_if_me,
                                                    want_now=False, want_vb=False, want_vbn=False)
                    source_subject_list = tokens[new_NP_index[0]:new_NP_index[1] + 1]
                    subject_list = real_token[new_NP_index[0]:new_NP_index[1] + 1]
                    source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him',
                                                                                                 'us']:
                        subject_list = [ob_sb(source_subject_list[0].text.lower())]
                    if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                        object_list = [ob_sb(source_object_list[0].text.lower())]
                    elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                        object_list[0] = object_list[0].lower()

                    final_sentence = real_token[:part_NP_indexes[0]]
                    final_sentence.extend(subject_list)
                    final_sentence.extend([new_vb])
                    final_sentence.extend(object_list)
                    final_sentence.extend(real_token[part_VP_indexes[0] + 2:new_NP_index[0] - 1])
                    final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                    final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                    new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                    if_changed = True
                    continue
                if tokens[part_VP_indexes[0]].lemma_ \
                        in ['can', 'could', 'may', 'might', 'must', 'will', 'should', 'would'] \
                        and tokens[part_VP_indexes[0] + 1].text == 'be' and \
                        tokens[part_VP_indexes[0] + 2].tag_ == 'VBN':
                    if real_token[part_VP_indexes[0] + 3] == 'and':
                        new_article.append(sentence.text)
                        continue
                    new_NP_index = [by_index + 1, part_VP_indexes[1]]
                    new_NP_if_single, new_NP_if_me = if_singular(tokens[new_NP_index[0]: new_NP_index[1]])
                    new_vb = tokens[part_VP_indexes[0] + 2].lemma_

                    source_subject_list = tokens[new_NP_index[0]:new_NP_index[1] + 1]
                    subject_list = real_token[new_NP_index[0]:new_NP_index[1] + 1]
                    source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him',
                                                                                                 'us']:
                        subject_list = [ob_sb(source_subject_list[0].text.lower())]
                    if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                        object_list = [ob_sb(source_object_list[0].text.lower())]
                    elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                        object_list[0] = object_list[0].lower()

                    final_sentence = real_token[:part_NP_indexes[0]]
                    final_sentence.extend(subject_list)
                    final_sentence.extend([tokens[part_VP_indexes[0]].lemma_, new_vb])
                    final_sentence.extend(object_list)
                    final_sentence.extend(real_token[part_VP_indexes[0] + 3:new_NP_index[0] - 1])
                    final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                    final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                    new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                    if_changed = True
                    continue
                if tokens[part_VP_indexes[0]].lemma_ == 'be' and tokens[part_VP_indexes[0] + 1].text == 'being' and \
                        tokens[part_VP_indexes[0] + 2].tag_ == 'VBN':
                    if real_token[part_VP_indexes[0] + 3] == 'and':
                        new_article.append(sentence.text)
                        continue
                    new_NP_index = [by_index + 1, part_VP_indexes[1]]
                    new_NP_if_single, new_NP_if_me = if_singular(tokens[new_NP_index[0]: new_NP_index[1]])
                    new_vb_be = corresponding_tense_vb('be',
                                                       if_now=new_NP_if_now,
                                                       if_singular=new_NP_if_single,
                                                       if_me=new_NP_if_me,
                                                       want_now=False, want_vb=False, want_vbn=False)
                    new_vb2 = corresponding_tense_vb(tokens[part_VP_indexes[0] + 2].lemma_,
                                                     if_now=new_NP_if_now,
                                                     if_singular=new_NP_if_single,
                                                     if_me=new_NP_if_me,
                                                     want_now=True, want_vb=False, want_vbn=False)
                    source_subject_list = tokens[new_NP_index[0]:new_NP_index[1] + 1]
                    subject_list = real_token[new_NP_index[0]:new_NP_index[1] + 1]
                    source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him',
                                                                                                 'us']:
                        subject_list = [ob_sb(source_subject_list[0].text.lower())]
                    if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                        object_list = [ob_sb(source_object_list[0].text.lower())]
                    elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                        object_list[0] = object_list[0].lower()

                    final_sentence = real_token[:part_NP_indexes[0]]
                    final_sentence.extend(subject_list)
                    final_sentence.extend([new_vb_be, new_vb2])
                    final_sentence.extend(object_list)
                    final_sentence.extend(real_token[part_VP_indexes[0] + 3:new_NP_index[0] - 1])
                    final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                    final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                    new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                    if_changed = True
                    continue
                if tokens[part_VP_indexes[0]].lemma_ == 'have' and tokens[part_VP_indexes[0] + 1].text == 'been' and \
                        tokens[part_VP_indexes[0] + 2].tag_ == 'VBN':
                    if real_token[part_VP_indexes[0] + 3] == 'and':
                        new_article.append(sentence.text)
                        continue
                    new_NP_index = [by_index + 1, part_VP_indexes[1]]
                    new_NP_if_single, new_NP_if_me = if_singular(tokens[new_NP_index[0]: new_NP_index[1]])
                    new_vb_have = corresponding_tense_vb('have',
                                                         if_now=new_NP_if_now,
                                                         if_singular=new_NP_if_single,
                                                         if_me=new_NP_if_me,
                                                         want_now=False, want_vb=False, want_vbn=False)
                    new_vb2 = corresponding_tense_vb(tokens[part_VP_indexes[0] + 2].lemma_,
                                                     if_now=new_NP_if_now,
                                                     if_singular=new_NP_if_single,
                                                     if_me=new_NP_if_me,
                                                     want_now=False, want_vb=False, want_vbn=True)
                    source_subject_list = tokens[new_NP_index[0]:new_NP_index[1] + 1]
                    subject_list = real_token[new_NP_index[0]:new_NP_index[1] + 1]
                    source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                    if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him',
                                                                                                 'us']:
                        subject_list = [ob_sb(source_subject_list[0].text.lower())]
                    if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                        object_list = [ob_sb(source_object_list[0].text.lower())]
                    elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                        object_list[0] = object_list[0].lower()

                    final_sentence = real_token[:part_NP_indexes[0]]
                    final_sentence.extend(subject_list)
                    final_sentence.extend([new_vb_have, new_vb2])
                    final_sentence.extend(object_list)
                    final_sentence.extend(real_token[part_VP_indexes[0] + 3:new_NP_index[0] - 1])
                    final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                    final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                    new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                    if_changed = True
                    continue
            if tokens[part_VP_indexes[0]].lemma_ in ['can', 'could', 'may', 'might', 'must', 'will', 'should', 'would']:
                part_VP_tag = [token.tag_ for token in tokens[part_VP_indexes[0]:part_VP_indexes[1] + 1]]
                if 'PRP' in part_VP_tag or 'PRP$' in part_VP_tag:
                    new_article.append(sentence.text)
                    continue
                if part_VP_indexes[1] - part_VP_indexes[0] < 2:
                    new_article.append(sentence.text)
                    continue
                if part_VP_indexes[1] - part_VP_indexes[0] > 2:
                    if tokens[part_VP_indexes[0]].lemma_ in ['can', 'could', 'may', 'might', 'must', 'will', 'should',
                                                             'would'] and tokens[
                        part_VP_indexes[0] + 1].text == 'have' and \
                            tokens[part_VP_indexes[0] + 2].tag_ == 'VBN':
                        if real_token[part_VP_indexes[0] + 3] == 'and' or tokens[part_VP_indexes[0] + 3].tag_ == 'IN':
                            new_article.append(sentence.text)
                            continue
                        source_subject_list = tokens[part_VP_indexes[0] + 3:part_VP_indexes[1] + 1]
                        subject_list = real_token[part_VP_indexes[0] + 3:part_VP_indexes[1] + 1]
                        source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                        object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                        if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them',
                                                                                                     'him', 'us']:
                            subject_list = [ob_sb(source_subject_list[0].text.lower())]
                        if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we',
                                                                                                   'he']:
                            object_list = [ob_sb(source_object_list[0].text.lower())]
                        elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                            object_list[0] = object_list[0].lower()

                        final_sentence = real_token[:part_NP_indexes[0]]
                        final_sentence.extend(subject_list)
                        final_sentence.extend(
                            [real_token[part_VP_indexes[0]], 'have', 'been', real_token[part_VP_indexes[0] + 2], 'by'])
                        final_sentence.extend(object_list)
                        final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                        final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                        new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                        if_changed = True
                        continue
                    if tokens[part_VP_indexes[0]].lemma_ in ['can', 'could', 'may', 'might', 'must', 'will', 'should',
                                                             'would'] \
                            and 'VB' in tokens[part_VP_indexes[0] + 1].tag_ \
                            and 'VB' not in tokens[part_VP_indexes[0] + 2].tag_:
                        tokens_dep = [token.dep_ for token in tokens]
                        if real_token[part_VP_indexes[0] + 2] == 'and' or 'dobj' not in tokens_dep \
                                or tokens[part_VP_indexes[0] + 2].tag_ == 'IN' \
                                or tokens[part_VP_indexes[0] + 1].lemma_ == 'be':
                            new_article.append(sentence.text)
                            continue
                        if 'there' in real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1] \
                                or 'There' in real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]:
                            new_article.append(sentence.text)
                            continue
                        new_vb = corresponding_tense_vb(tokens[part_VP_indexes[0] + 1].lemma_,
                                                        if_now=False,
                                                        if_singular=False,
                                                        if_me=False,
                                                        want_now=False, want_vb=False, want_vbn=True)
                        source_subject_list = tokens[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                        subject_list = real_token[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                        source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                        object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                        if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them',
                                                                                                     'him', 'us']:
                            subject_list = [ob_sb(source_subject_list[0].text.lower())]
                        if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we',
                                                                                                   'he']:
                            object_list = [ob_sb(source_object_list[0].text.lower())]
                        elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                            object_list[0] = object_list[0].lower()

                        final_sentence = real_token[:part_NP_indexes[0]]
                        final_sentence.extend(subject_list)
                        final_sentence.extend([real_token[part_VP_indexes[0]], 'be', new_vb, 'by'])
                        final_sentence.extend(object_list)
                        final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                        final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                        new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                        if_changed = True
                        continue
            if tokens[part_VP_indexes[0]].lemma_ == 'be' and tokens[part_VP_indexes[0] + 1].tag_ == 'VBG':
                part_VP_tag = [token.tag_ for token in tokens[part_VP_indexes[0]:part_VP_indexes[1] + 1]]
                if 'PRP' in part_VP_tag or 'PRP$' in part_VP_tag:
                    new_article.append(sentence.text)
                    continue
                if part_VP_indexes[1] - part_VP_indexes[0] < 2:
                    new_article.append(sentence.text)
                    continue
                tokens_dep = [token.dep_ for token in tokens]
                if real_token[part_VP_indexes[0] + 2] == 'and' or 'dobj' not in tokens_dep \
                        or tokens[part_VP_indexes[0] + 2].tag_ == 'IN' \
                        or tokens[part_VP_indexes[0] + 1].lemma_ == 'be':
                    new_article.append(sentence.text)
                    continue
                new_NP_if_single, new_NP_if_me = if_singular(tokens[part_VP_indexes[0]: part_VP_indexes[1] + 1])
                new_vb = corresponding_tense_vb('be',
                                                if_now=new_NP_if_now,
                                                if_singular=new_NP_if_single,
                                                if_me=new_NP_if_me,
                                                want_now=False, want_vb=False, want_vbn=False)
                new_vb_VBN = corresponding_tense_vb(tokens[part_VP_indexes[0] + 1].lemma_,
                                                    if_now=False,
                                                    if_singular=False,
                                                    if_me=False,
                                                    want_now=False, want_vb=False, want_vbn=True)
                source_subject_list = tokens[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                subject_list = real_token[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him', 'us']:
                    subject_list = [ob_sb(source_subject_list[0].text.lower())]
                if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                    object_list = [ob_sb(source_object_list[0].text.lower())]
                elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                    object_list[0] = object_list[0].lower()

                final_sentence = real_token[:part_NP_indexes[0]]
                final_sentence.extend(subject_list)
                final_sentence.extend([new_vb, 'being', new_vb_VBN, 'by'])
                final_sentence.extend(object_list)
                final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                if_changed = True
                continue
            if tokens[part_VP_indexes[0]].lemma_ == 'have' and tokens[part_VP_indexes[0] + 1].tag_ == 'VBN':
                part_VP_tag = [token.tag_ for token in tokens[part_VP_indexes[0]:part_VP_indexes[1] + 1]]
                if 'PRP' in part_VP_tag or 'PRP$' in part_VP_tag:
                    new_article.append(sentence.text)
                    continue
                if part_VP_indexes[1] - part_VP_indexes[0] < 2:
                    new_article.append(sentence.text)
                    continue
                if real_token[part_VP_indexes[0]:part_VP_indexes[0] + 2] == ['taken', 'place']:
                    new_article.append(sentence.text)
                    continue
                tokens_dep = [token.dep_ for token in tokens]
                if real_token[part_VP_indexes[0] + 2] == 'and' or 'dobj' not in tokens_dep \
                        or tokens[part_VP_indexes[0] + 2].tag_ == 'IN' \
                        or tokens[part_VP_indexes[0] + 1].lemma_ == 'be':
                    new_article.append(sentence.text)
                    continue
                new_NP_if_single, new_NP_if_me = if_singular(tokens[part_VP_indexes[0]: part_VP_indexes[1] + 1])
                new_vb = corresponding_tense_vb('have',
                                                if_now=new_NP_if_now,
                                                if_singular=new_NP_if_single,
                                                if_me=new_NP_if_me,
                                                want_now=False, want_vb=False, want_vbn=False)
                new_vb_VBN = corresponding_tense_vb(tokens[part_VP_indexes[0] + 1].lemma_,
                                                    if_now=False,
                                                    if_singular=False,
                                                    if_me=False,
                                                    want_now=False, want_vb=False, want_vbn=True)
                source_subject_list = tokens[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                subject_list = real_token[part_VP_indexes[0] + 2:part_VP_indexes[1] + 1]
                source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him', 'us']:
                    subject_list = [ob_sb(source_subject_list[0].text.lower())]
                if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                    object_list = [ob_sb(source_object_list[0].text.lower())]
                elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                    object_list[0] = object_list[0].lower()

                final_sentence = real_token[:part_NP_indexes[0]]
                final_sentence.extend(subject_list)
                final_sentence.extend([new_vb, 'been', new_vb_VBN, 'by'])
                final_sentence.extend(object_list)
                final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                if_changed = True
                continue
            if 'VB' in tokens[part_VP_indexes[0]].tag_ and tokens[part_VP_indexes[0]].lemma_ != 'be':
                part_VP_tag = [token.tag_ for token in tokens[part_VP_indexes[0]:part_VP_indexes[1] + 1]]
                if 'PRP' in part_VP_tag or 'PRP$' in part_VP_tag:
                    new_article.append(sentence.text)
                    continue
                if part_VP_indexes[1] - part_VP_indexes[0] < 1:
                    new_article.append(sentence.text)
                    continue
                if tokens[part_VP_indexes[0]].lemma_ == 'take' and real_token[part_VP_indexes[0] + 1] == 'place':
                    new_article.append(sentence.text)
                    continue
                tokens_dep = [token.dep_ for token in tokens]
                if real_token[part_VP_indexes[0] + 1] == 'and' or 'dobj' not in tokens_dep \
                        or tokens[part_VP_indexes[0] + 1].tag_ == 'IN' \
                        or tokens[part_VP_indexes[0]].lemma_ == 'be':
                    new_article.append(sentence.text)
                    continue
                new_NP_if_single, new_NP_if_me = if_singular(tokens[part_VP_indexes[0]: part_VP_indexes[1] + 1])
                new_vb = corresponding_tense_vb('be',
                                                if_now=new_NP_if_now,
                                                if_singular=new_NP_if_single,
                                                if_me=new_NP_if_me,
                                                want_now=False, want_vb=False, want_vbn=False)
                new_vb_VBN = corresponding_tense_vb(tokens[part_VP_indexes[0]].lemma_,
                                                    if_now=False,
                                                    if_singular=False,
                                                    if_me=False,
                                                    want_now=False, want_vb=False, want_vbn=True)
                source_subject_list = tokens[part_VP_indexes[0] + 1:part_VP_indexes[1] + 1]
                subject_list = real_token[part_VP_indexes[0] + 1:part_VP_indexes[1] + 1]
                source_object_list = tokens[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                object_list = real_token[part_NP_indexes[0]:part_NP_indexes[1] + 1]
                if len(source_subject_list) == 1 and source_subject_list[0].text.lower() in ['me', 'them', 'him', 'us']:
                    subject_list = [ob_sb(source_subject_list[0].text.lower())]
                if len(source_object_list) == 1 and source_object_list[0].text.lower() in ['i', 'they', 'we', 'he']:
                    object_list = [ob_sb(source_object_list[0].text.lower())]
                elif source_object_list[0].tag_ not in ['NNP', 'NNPS']:
                    object_list[0] = object_list[0].lower()

                final_sentence = real_token[:part_NP_indexes[0]]
                final_sentence.extend(subject_list)
                final_sentence.extend([new_vb, new_vb_VBN, 'by'])
                final_sentence.extend(object_list)
                final_sentence.extend(real_token[part_VP_indexes[1] + 1:])
                final_sentence[0] = final_sentence[0][0].upper() + final_sentence[0][1:]
                new_article.append(TreebankWordDetokenizer().detokenize(final_sentence))
                if_changed = True
                continue
            new_article.append(sentence.text)
            continue
        if if_changed:
            final_article = ''
            for sent in new_article:
                final_article = final_article + " " + sent
            final_article = final_article[1:]
            return question, final_article, True
        else:
            return question, article, False