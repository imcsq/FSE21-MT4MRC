import json
from MR1_1 import MR1_1
from MR1_2 import MR1_2
from MR1_3 import MR1_3
from MR1_4 import MR1_4
from MR2_1 import MR2_1
from MR2_2 import MR2_2
from MR2_3 import MR2_3

class MRs(object):

    def generate(self, data, mr):
        this_line = json.loads(data.strip())
        idx = this_line["idx"]
        question = this_line["question"]
        passage = this_line["passage"]
        label = this_line["label"]
        assert mr in ["1_1", "1_2", "1_3", "1_4", "2_1", "2_2", "2_3"]
        if mr == "1_1":
            this_mr = MR1_1()
        if mr == "1_2":
            this_mr = MR1_2()
        if mr == "1_3":
            this_mr = MR1_3()
        if mr == "1_4":
            this_mr = MR1_4()
        if mr == "2_1":
            this_mr = MR2_1()
        if mr == "2_2":
            this_mr = MR2_2()
        if mr == "2_3":
            this_mr = MR2_3()
        followup_question, followup_article, if_eligible = this_mr.generate(question, passage)
        if if_eligible:
            source_data = "{\"idx\": "+idx+", \"question\": "+question+", \"passage\""+passage+", \"label\""+label+"}"
            follow_data = "{\"idx\": "+idx+", \"question\": "+followup_question+", \"passage\""+followup_article+", \"label\""+label+"}"
            return [source_data, follow_data]
        else:
            return None