import pandas as pd

from graphs.parsers.CompoundRebusGraphParser import CompoundRebusGraphParser
from graphs.patterns.Rule import Rule

compounds = pd.read_csv("./saved/ladec_raw.csv")

Rule().check_homophones("to")
Rule().check_homophones("for")

# parser = CompoundRebusGraphParser()
# graphs = parser.parse("to", "for", False)
# print(graphs[0])
