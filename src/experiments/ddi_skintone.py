import dspy
import pandas as pd
from PIL import Image
import os
from .base import BaseExperiment
from ..metrics import exact_match
from config.paths import PATHS


class DDISkinTone(dspy.Signature):
    """Based on this image, pretend you are a dermatologist and identify the patient's Fitzpatrick skin tone group. Use only the provided list of skin tone groups."""
    skin_image: dspy.Image = dspy.InputField()
    choices: list = dspy.InputField(prefix="Choices:", desc="List of comma-separated choices to choose from")
    skin_tone: str = dspy.OutputField()


class DDISkintoneExperiment(BaseExperiment):
    def run(self):
        choices = ["12", "34", "56"]
        def load_ddi_examples(csv_path, image_folder):
            df = pd.read_csv(csv_path)
            def load_image(filename):
                path = os.path.join(image_folder, filename)
                return Image.open(path).convert("RGB")
            df["image"] = df["DDI_file"].apply(load_image)
            examples = []
            for _, row in df.iterrows():
                ex = dspy.Example(
                    skin_image=dspy.Image.from_PIL(row["image"]),
                    choices = choices,
                    ground_truth_skin_tone=row["skin_tone"],
                    ground_truth_malignant=row["malignant"],
                    ground_truth_disease=row["disease"]
                ).with_inputs("skin_image", "choices")
                examples.append(ex)
            return examples
        
        testset = load_ddi_examples(str(PATHS["ddi"]["test"]), str(PATHS["ddi"]["images"]))
        trainset = load_ddi_examples(str(PATHS["ddi"]["train"]), str(PATHS["ddi"]["images"]))
        
        DDISkinToneProgram = dspy.ChainOfThought(DDISkinTone)
        def metric(example, pred, trace=None) -> int:
            return exact_match(str(example.ground_truth_skin_tone), str(pred.skin_tone))
        return trainset, testset, DDISkinToneProgram, metric


def run_ddi_skintone():
    experiment = DDISkintoneExperiment()
    return experiment.run() 