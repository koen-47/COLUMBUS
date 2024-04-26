import json
import re

import pandas as pd

with open("../saved/theidioms_raw.json", "r") as file:
    idioms = pd.DataFrame(json.load(file))

for idiom in idioms["idiom"]:
    # idiom_ = re.sub(" the | a |^(a )|\(.*?\)| of ", " ", idiom).strip()
    # # print(idiom_)
    # # idiom_ = re.split(" in | into | inside ", idiom_)
    # # idiom_ = re.split(" over | on ", idiom_)
    # # idiom_ = re.split(" out |^(out) ", idiom_)
    # # idiom_ = re.split(" down |^(down) ", idiom_)
    # idiom_ = re.split(" far |^(far) ", idiom_)
    # if len(idiom_) > 1:
    #     print(idiom, idiom_)
    if len(idiom.split()) == 2:
        print(idiom)
