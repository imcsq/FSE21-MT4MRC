import json
import numpy
from MR1_1 import MR1_1
from MR1_2 import MR1_2
from MR1_3 import MR1_3
from MR1_4 import MR1_4
from MR2_1 import MR2_1
from MR2_2 import MR2_2
from MR2_3 import MR2_3


class MRs(object):

    def generate(self, data, mr):
        if mr == "1_1":
            this_mr = MR1_1()
            return this_mr.generate(data)
        if mr == "1_2":
            this_mr = MR1_2()
            return this_mr.generate(data)
        if mr == "1_3":
            this_mr = MR1_3()
            return this_mr.generate(data)
        if mr == "1_4":
            this_mr = MR1_4()
            return this_mr.generate(data)
        if mr == "2_1":
            this_mr = MR2_1()
            return this_mr.generate(data)
        if mr == "2_2":
            this_mr = MR2_2()
            return this_mr.generate(data)
        if mr == "2_3":
            this_mr = MR2_3()
            return this_mr.generate(data)

