# This source file is part of the Daneshjou Lab project
#
# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see AUTHORS.md)
#
# SPDX-License-Identifier: MIT

import dspy
from datasets import load_dataset
from .base import BaseExperiment
from ..metrics import quasi_exact_match


class VQA_Radiology(dspy.Signature):
    """
    Given the radiology image, answer the question using a single word or sentence.
    """
    radiology_image: dspy.Image = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class VQARadExperiment(BaseExperiment):
    def run(self):
        HUGGING_FACE_DATASET_PATH = "flaviagiammarino/vqa-rad"
        def process_example(example):
            return dspy.Example(
                question=example["question"].strip(),
                radiology_image=dspy.Image.from_PIL(example["image"]),
                ground_truth_answer=example["answer"].strip()
            ).with_inputs("question", "radiology_image")
        def get_instances(split):
            dataset = load_dataset(HUGGING_FACE_DATASET_PATH, split=split)
            return [process_example(ex) for ex in dataset]
        trainset = get_instances("train")[:500]
        testset = get_instances("test")

        VQA_RadiologyProgram = dspy.ChainOfThought(VQA_Radiology)
        def metric(example, pred, trace=None):
            is_correct = quasi_exact_match(example.ground_truth_answer, pred.answer)
            feedback_text = "Correct prediction." if is_correct else f"Incorrect prediction. Expected: {example.ground_truth_answer}"
            return dspy.Prediction(
                score=is_correct,
                feedback=feedback_text
            )
        return trainset, testset, VQA_RadiologyProgram, metric


def run_vqa_rad():
    experiment = VQARadExperiment()
    return experiment.run()