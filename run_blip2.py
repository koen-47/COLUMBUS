import os

from scripts.models.BLIP2Experiment import BLIP2Experiment

blip2 = BLIP2Experiment(size="6.7b")
blip2.run_on_benchmark(f"{os.path.dirname(__file__)}/results/experiments")

# fuyu = FuyuExperiment()
# fuyu.run_on_benchmark(f"{os.path.dirname(__file__)}/results/experiments")
