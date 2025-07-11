#!/usr/bin/env python3
import sys
import os
import subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

EXPERIMENTS = ["vqa_rad", "chexpert", "ddi_disease", "ddi_skintone", "gastrovision"]

def run_batch_experiments(model, api_base, api_key, experiments=None, cache=True, cache_dir=None):
    if experiments is None:
        experiments = EXPERIMENTS
    
    for exp in experiments:
        print(f"Running experiment: {exp}")
        cmd = [
            "python", os.path.join(os.path.dirname(__file__), "run_experiment.py"),
            "--experiment", exp,
            "--model", model,
            "--api_base", api_base,
            "--api_key", api_key,
            "--cache", str(cache)
        ]
        
        if cache_dir:
            cmd.extend(["--cache_dir", cache_dir])
        
        try:
            subprocess.run(cmd, check=True)
            print(f"Completed experiment: {exp}")
        except subprocess.CalledProcessError as e:
            print(f"Failed experiment: {exp} with error: {e}")
            continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_base", required=True)
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--experiments", nargs="+", default=EXPERIMENTS)
    parser.add_argument("--cache", default=True, type=bool)
    parser.add_argument("--cache_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    run_batch_experiments(
        args.model, 
        args.api_base, 
        args.api_key, 
        args.experiments, 
        args.cache, 
        args.cache_dir
    ) 