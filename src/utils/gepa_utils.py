import dspy
from dspy.teleprompt import GEPA


class ProposeNewInstructionSignature(dspy.Signature):
    """Your task is to **create a comprehensive, clear, and fully self-contained instruction document** that enables people with no prior experience to successfully perform a complex task. The instructions must ensure that users can complete the task accurately using only the document you produce without requiring any external sources.
---
## What You Will Receive
You will be provided with:
- **Original Instructions:**
The previous version of the instruction document.
- **User Examples and Feedback:**
Outputs from users of the previous instruction document, with notes highlighting common mistakes and misunderstandings stemming from incompleteness of the previous guidance document as feedback.
---
## What You Must Produce
Develop a **detailed instruction document** that:
- **Is self-contained.**
All needed definitions, rules, and context are included. Do not refer to any external documents or sources.
- **Addresses all prior confusion.**
Integrate explicit corrections based on user feedback and examples of past errors.
- **Uses clear, precise language.**
- **Includes practical examples.**
Show both correct and incorrect task execution, with explanations.
- **Explains all key terms and concepts.**
- **States all requirements.**
Clearly specify any output formatting requirements, acceptable outputs, and evaluation criteria.
- **Covers edge cases and exceptions.**
- **Includes all required domain knowledge from feedbacks and reference material**
- **Includes repeatable task patterns in how previous users attempted the task correctly**
- **Do not waste space in formatting and headers of the document. Be direct.**
Follow these directions to produce instructions that enable any user—without any prior knowledge of task domain—to complete the task successfully and consistently."""

    current_instructions: str = dspy.InputField(
        desc="The current (previous) version of the instruction document that requires improvement."
    )
    
    examples_summary: str = dspy.InputField(
        desc="User-generated example outputs and corresponding feedback, indicating recurring mistakes, misunderstandings, and any confusion about previous instructions."
    )
    
    improved_instructions: str = dspy.OutputField(
        desc="A comprehensive, self-contained, and clear instruction document that resolves all identified user issues, defines all necessary concepts, provides explicit examples (correct and incorrect with explanations), specifies all requirements, and incorporates all provided domain and feedback-informed guidance."
    )


class ImageInstructionProposer:
    def __call__(
        self, 
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict]], 
        components_to_update: list[str]
    ) -> dict[str, str]:
        new_texts = {}
        
        for name in components_to_update:
            current_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]
            
            formatted_samples = "\n".join([
                f"Input: {item.get('inputs', '')}\n"
                f"Output: {item.get('outputs', '')}\n"
                f"Feedback: {item.get('feedback', '')}"
                for item in dataset_with_feedback
            ])

            propose_module = dspy.Predict(ProposeNewInstructionSignature)
            result = propose_module(
                current_instructions=current_instruction,
                examples_summary=formatted_samples
            )
            
            new_texts[name] = result.improved_instructions
        
        return new_texts


def create_gepa_teleprompter(metric, lm, trainset):
    teleprompter = GEPA(
        metric=metric,
        num_threads=15,
        track_stats=True,
        use_merge=False,
        reflection_lm=lm,
        max_metric_calls=40 * len(trainset),
        instruction_proposer=ImageInstructionProposer()
    )
    return teleprompter