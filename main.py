import argparse

from results.analysis.AnalysisReport import AnalysisReport
from results.benchmark.PuzzleAnalysisReport import PuzzleAnalysisReport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=str)
    parser.add_argument("--run", type=str, default="overall")

    args = parser.parse_args()
    analysis_type = args.analysis

    if analysis_type == "puzzles":
        puzzle_analysis = PuzzleAnalysisReport()
        puzzle_analysis.generate()
    elif analysis_type == "models":
        model_analysis = AnalysisReport(args.run)
        model_analysis.generate_all()


if __name__ == "__main__":
    main()
