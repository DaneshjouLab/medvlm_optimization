import argparse
import os
import dspy
import logging
import sys
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, MIPROv2, SIMBA
from utils.gepa_utils import create_gepa_teleprompter

from experiments.vqa_rad import run_vqa_rad
from experiments.chexpert import run_chexpert
from experiments.ddi_disease import run_ddi_disease
from experiments.ddi_skintone import run_ddi_skintone
from experiments.gastrovision import run_gastrovision
from utils.logging import setup_logging

logging.getLogger("dspy.utils.parallelizer").setLevel(logging.CRITICAL)
logging.getLogger("dspy.adapters.json_adapter").setLevel(logging.CRITICAL)
logging.getLogger("dspy.teleprompt.bootstrap").setLevel(logging.CRITICAL)

EXPERIMENTS = {
    "vqa_rad": run_vqa_rad,
    "chexpert": run_chexpert,
    "ddi_disease": run_ddi_disease,
    "ddi_skintone": run_ddi_skintone,
    "gastrovision": run_gastrovision
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS.keys(), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--cache", default=True, type=bool)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    if args.cache_dir is None:
        cache_dir = os.path.expanduser(f"~/.dspy_cache/{args.experiment}_{model_short}")
    else:
        cache_dir = args.cache_dir

    print('THIS IS CACHE DIR!')
    print(cache_dir)

    dspy.configure_cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=cache_dir
    )

    exp_dir = args.experiment
    log_filename = setup_logging(exp_dir, args.experiment, args)

    print(f"Arguments: {args}")
    print(f"Logging all output to {log_filename}\n")

    lm = dspy.LM(model=args.model, api_base=args.api_base, api_key=args.api_key, cache=args.cache)
    dspy.configure(lm=lm)

    trainset, testset, program, metric = EXPERIMENTS[args.experiment]()
    evaluate_program = Evaluate(devset=testset, metric=metric, num_threads=8, display_progress=True, max_errors=5000)
    evaluate_program(program)

    optimizer = BootstrapFewShotWithRandomSearch(metric=metric, num_candidate_programs=10, max_errors=5000)
    compiled_program = optimizer.compile(program, trainset=trainset)
    evaluate_program(compiled_program)

    config = dict(num_candidates=19,  num_threads=10)
    optimizer = MIPROv2(metric=metric, max_errors=5000, **config, auto=None)
    compiled_program = optimizer.compile(program, trainset=trainset, requires_permission_to_run=False, num_trials=25)
    evaluate_program(compiled_program)

    teleprompter = SIMBA(metric=metric)
    compiled_program = teleprompter.compile(program, trainset=trainset, seed=6793115)
    evaluate_program(compiled_program)

    teleprompter = create_gepa_teleprompter(metric, lm, trainset)
    compiled_program = teleprompter.compile(program, trainset=trainset, valset=trainset)
    evaluate_program(compiled_program)


if __name__ == "__main__":
    main() 