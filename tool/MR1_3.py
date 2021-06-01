import json

class MR1_3(object):

    def generate(self, data):
        if "label" not in data:
            data = data[:len(data) - 2] + ", \"label\": true}"
        output = []
        line = data
        this_line = json.loads(line.strip("\n"))
        question = this_line["question"]
        if " before " in this_line["question"]:
            question_2 = str(question).replace(" before ", " after ", 1)
            out_line = str(line).replace(question, question_2, 1)
            output.append([line.strip(), out_line.strip()])
        elif " after " in this_line["question"]:
            question_2 = str(question).replace(" after ", " before ", 1)
            out_line = str(line).replace(question, question_2, 1)
            output.append([line.strip(), out_line.strip()])
        return  output