import dspy
import pandas as pd
from PIL import Image
import os
from .base import BaseExperiment
from config.paths import PATHS


class DDISkinTone(dspy.Signature):
    """Based on this image, pretend you are a dermatologist and list your top 3 most likely diagnoses in order of concern. Use only the provided list of diagnosis choices."""
    skin_image: dspy.Image = dspy.InputField()
    choices: list = dspy.InputField(prefix="Choices:", desc="List of comma-separated choices to choose from")
    diagnosis_choices: list = dspy.OutputField(desc="Only top 3 diagnoses in order of concern")


class DDIDiseaseExperiment(BaseExperiment):
    def run(self):
        choices = [
            "melanoma-in-situ", "mycosis-fungoides", "squamous-cell-carcinoma-in-situ",
            "basal-cell-carcinoma", "squamous-cell-carcinoma", "melanoma-acral-lentiginous",
            "basal-cell-carcinoma-superficial", "squamous-cell-carcinoma-keratoacanthoma",
            "subcutaneous-t-cell-lymphoma", "melanocytic-nevi", "seborrheic-keratosis-irritated",
            "focal-acral-hyperkeratosis", "hyperpigmentation", "lipoma", "foreign-body-granuloma",
            "blue-nevus", "verruca-vulgaris", "acrochordon", "wart", "epidermal-nevus",
            "abrasions-ulcerations-and-physical-injuries", "basal-cell-carcinoma-nodular",
            "epidermal-cyst", "acquired-digital-fibrokeratoma",
            "seborrheic-keratosis", "trichilemmoma", "pyogenic-granuloma", "neurofibroma",
            "syringocystadenoma-papilliferum", "nevus-lipomatosus-superficialis", "benign-keratosis",
            "inverted-follicular-keratosis", "onychomycosis", "dermatofibroma", "trichofolliculoma",
            "lymphocytic-infiltrations", "prurigo-nodularis", "kaposi-sarcoma", "scar", "eccrine-poroma",
            "angioleiomyoma", "keloid", "hematoma", "metastatic-carcinoma", "melanoma", "angioma",
            "folliculitis", "atypical-spindle-cell-nevus-of-reed", "xanthogranuloma",
            "eczema-spongiotic-dermatitis", "arteriovenous-hemangioma", "acne-cystic",
            "verruciform-xanthoma", "molluscum-contagiosum", "condyloma-accuminatum", "morphea",
            "neuroma", "dysplastic-nevus", "nodular-melanoma-(nm)", "actinic-keratosis",
            "pigmented-spindle-cell-nevus-of-reed", "dermatomyositis", "glomangioma",
            "cellular-neurothekeoma", "fibrous-papule", "graft-vs-host-disease", "lichenoid-keratosis",
            "reactive-lymphoid-hyperplasia", "coccidioidomycosis", "leukemia-cutis",
            "sebaceous-carcinoma", "chondroid-syringoma", "tinea-pedis", "solar-lentigo",
            "clear-cell-acanthoma", "abscess", "blastic-plasmacytoid-dendritic-cell-neoplasm",
            "acral-melanotic-macule"
        ]
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
            return example.ground_truth_disease in pred.diagnosis_choices[:3]
        return trainset, testset, DDISkinToneProgram, metric


def run_ddi_disease():
    experiment = DDIDiseaseExperiment()
    return experiment.run() 