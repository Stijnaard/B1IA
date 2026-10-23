from CI_project import LLM, Dataset, construct_dataset, PromptGenerator, ForwardFlower 
from pathlib import Path
import pytest

@pytest.mark.acceptance
def test_walking_skeleton():
    # 1. instantiation:
    model: LLM = LLM("liquid/lfm2.5-1.2b", "http://localhost:1234/v1")
    dataset_folder: Path = Path(".") / "datasets"
    remaining_attempts: int = 2
    satisfying_summary: bool = False

    # 2. construct the christmas data:
    christmas_data: Dataset = construct_dataset(dataset_folder = dataset_folder, topic="christmas")
    
    # 3. construct the prompt generators:
    summary_prompt_generator = PromptGenerator()
    revision_prompt_generator = PromptGenerator()

    # 4. construct the forward mover:
    forward_mover = ForwardFlower(model,
                                christmas_data,
                                summary_prompt_generator,
                                revision_prompt_generator)
    
    # 5. keep trying to construct the best summary until no more attempts, or satisfying summary:
    while remaining_attempts != 0 and not satisfying_summary:
        generated_summary, inconsistent_queries, undetermined_queries  = forward_mover.forward()

    assert generated_summary == "fake summary #1"
    assert inconsistent_queries == ['inconsistent query 1', 'inconsistent query 2']
