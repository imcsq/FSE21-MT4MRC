import json
import spacy
import nltk
from nltk.corpus import wordnet
nlp = spacy.load('en_core_web_sm')


class MR2_1(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        line = data
        this_line = json.loads(line.strip("\n"))
        question = this_line["question"]
        word = []
        place = []
        num = 0
        doc = nlp(question)
        tokens = [token for token in doc]
        for token in tokens:
            if token.tag_ in ["JJ"]:
                word.append(str(token.string.strip()))
                place.append(num)
            num += 1
        rd = -1
        output_words = []
        for one in word:
            rd += 1
            synonyms = []
            for syn in wordnet.synsets(one, "a"):
                for lemma in syn.lemmas():
                    synonyms.append(lemma.name().lower())
            if (len(synonyms) == 0):
                continue
            if place[rd] == (len(tokens) - 1):
                for num in range(len(synonyms)):
                    if synonyms[num] == one:
                        continue
                    else:
                        if [" " + one, " " + synonyms[num]] in output_words:
                            continue
                        output_words.append([" " + one, " " + synonyms[num]])
            else:
                for num in range(len(synonyms)):
                    if synonyms[num] == one:
                        continue
                    else:
                        if [" " + one + " ", " " + synonyms[num] + " "] in output_words:
                            continue
                        output_words.append([" " + one + " ", " " + synonyms[num] + " "])
        if output_words == []:
            return
        output = []
        for one in output_words:
            question_out = question
            question_out = question_out.replace(one[0], one[1], 1)
            out_line = line
            out_line = out_line.replace(question, question_out, 1)
            if [line.strip(), out_line.strip()] not in output:
                output.append([line.strip(), out_line.strip()])
        if output == []:
            return
        return output



