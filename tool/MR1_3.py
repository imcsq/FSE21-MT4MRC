import spacy
from nltk.tokenize.treebank import TreebankWordDetokenizer
nlp = spacy.load('en_core_web_sm')


class MR1_3(object):

    def generate(self, question, article):
        doc = nlp(question)
        tokens = [token for token in doc]
        real_tokens = [token.text for token in tokens]
        if 'before' in real_tokens:
            before_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'before':
                    before_index = index
                    break
            follow_question = real_tokens[:before_index]
            follow_question.append('after')
            if before_index != len(real_tokens)-1:
                follow_question.extend(real_tokens[before_index+1:])
            this_follow_question = TreebankWordDetokenizer().detokenize(follow_question)
            return this_follow_question, article, True
        elif 'after' in real_tokens:
            after_index = 0
            for index in range(len(real_tokens)):
                if real_tokens[index] == 'after':
                    after_index = index
                    break
            follow_question = real_tokens[:after_index]
            follow_question.append('before')
            if after_index != len(real_tokens)-1:
                follow_question.extend(real_tokens[after_index+1:])
            this_follow_question = TreebankWordDetokenizer().detokenize(follow_question)
            return this_follow_question, article, True
        return question, article, False