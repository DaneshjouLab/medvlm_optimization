# This source file is part of the Daneshjou Lab project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

import dspy
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
from .base import BaseExperiment
from ..metrics import f1_score
from config.paths import PATHS


class Gastrovision(dspy.Signature):
    """You are an AI doctor specializing in Gastroenterology. You are given a frame
    from a patient's endoscopy procedure and a list of possible diagnosis choices.
    Select only the single correct choice, and give the answer as a short response.
    If none of the choices is correct, output none. Do not explain.
    """
    endoscopy_procedure_image: dspy.Image = dspy.InputField()
    choices: list = dspy.InputField(prefix="Choices:", desc="List of comma-separated choices to choose from")
    diagnosis_choice: str = dspy.OutputField(desc="the single correct diagnosis choice from the list")


class GastrovisionExperiment(BaseExperiment):
    def run(self):
        choices = [
            "Accessory tools", "Angiectasia", "Barrett's esophagus", "Blood in lumen",
            "Cecum", "Colon diverticula", "Colon polyps", "Colorectal cancer", "Duodenal bulb",
            "Dyed-lifted-polyps", "Dyed-resection-margins", "Erythema", "Esophageal varices",
            "Esophagitis", "Gastric polyps", "Gastroesophageal_junction_normal z-line",
            "Ileocecal valve", "Mucosal inflammation large bowel", "Normal esophagus",
            "Normal mucosa and vascular pattern in the large bowel", "Normal stomach",
            "Pylorus", "Resected polyps", "Resection margins", "Retroflex rectum",
            "Small bowel terminal ileum", "Ulcer"
        ]
        def load_dspy_examples(csv_path):
            df = pd.read_csv(csv_path)
            examples = []
            for _, row in df.iterrows():
                base64_str = row["image_url"].split(",")[1]
                image = Image.open(BytesIO(base64.b64decode(base64_str)))
                ex = dspy.Example(
                    endoscopy_procedure_image=dspy.Image.from_PIL(image),
                    choices = choices,
                    ground_truth_diagnosis_choice=row["Class"]
                ).with_inputs("endoscopy_procedure_image", "choices")
                examples.append(ex)
            return examples

        trainset = load_dspy_examples(str(PATHS["gastrovision"]["train"]))
        testset = load_dspy_examples(str(PATHS["gastrovision"]["test"]))

        GastrovisionProgram = dspy.ChainOfThought(Gastrovision)
        def metric(example, pred, trace=None):
            score = f1_score(pred.diagnosis_choice, example.ground_truth_diagnosis_choice)
            is_correct = score > 0.8 if trace is not None else score == 1.0
            if is_correct:
                feedback_text = "Correct prediction."
            else:
                feedback_text = f"Incorrect prediction. Expected: {example.ground_truth_diagnosis_choice}"
            return dspy.Prediction(
                score=is_correct,
                feedback=feedback_text
            )
        return trainset, testset, GastrovisionProgram, metric


def run_gastrovision():
    experiment = GastrovisionExperiment()
    return experiment.run()