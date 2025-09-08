import dspy
import pandas as pd
from PIL import Image
from .base import BaseExperiment
from ..metrics import f1_score
from config.paths import PATHS


class Chexpert(dspy.Signature):
    """You are an AI doctor specializing in radiology. You are given the patient's chest
    radiograph and a list of possible diagnosis choices. Select all the correct choice(s),
    and give the answer as a short response. If none of the choices is correct, output none.
    Do not explain.
    """
    chest_radiograph_image: dspy.Image = dspy.InputField()
    choices: list = dspy.InputField(prefix="Choices:", desc="List of comma-separated choices to choose from")
    diagnosis_choice: str = dspy.OutputField(desc="the single correct diagnosis choice from the list")


class ChexpertExperiment(BaseExperiment):
    def run(self):
        choices = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
        def load_dspy_examples(csv_path):
            df = pd.read_csv(csv_path)
            examples = []
            if "image_path" in df.columns:
                for _, row in df.iterrows():
                    image = Image.open(row["image_path"]).convert("L")
                    ex = dspy.Example(
                        chest_radiograph_image=dspy.Image.from_PIL(image),
                        choices=choices,
                        ground_truth_diagnosis_choice=row["label"]
                    ).with_inputs("chest_radiograph_image", "choices")
                    examples.append(ex)
            elif "image_datauri" in df.columns:
                for _, row in df.iterrows():
                    ex = dspy.Example(
                        chest_radiograph_image=dspy.Image(url=row["image_datauri"]),
                        choices=choices,
                        ground_truth_diagnosis_choice=row["label"]
                    ).with_inputs("chest_radiograph_image", "choices")
                    examples.append(ex)
            return examples
        trainset = load_dspy_examples(str(PATHS["chexpert"]["train"]))
        testset = load_dspy_examples(str(PATHS["chexpert"]["test"]))
        
        ChexpertProgram = dspy.ChainOfThought(Chexpert)
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
        return trainset, testset, ChexpertProgram, metric


def run_chexpert():
    experiment = ChexpertExperiment()
    return experiment.run() 