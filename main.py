from graphs.parsers.PhraseRebusGraphParser import PhraseRebusGraphParser
from graphs.RebusImageConverterV2 import RebusImageConverterV2
from results.analysis.AnalysisReport import AnalysisReport
from results.benchmark.PuzzleAnalysisReport import PuzzleAnalysisReport


# AnalysisReport().generate(model_type="llava-1.6-34b", prompt_type=2, verbose=True)
# AnalysisReport().generate_all(verbose=True)

PuzzleAnalysisReport().generate_final()
PuzzleAnalysisReport().compute_basic_statistics()
