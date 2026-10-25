from CI_project import LLM, Dataset, construct_dataset, Agent, ForwardFlower
from pathlib import Path
import pytest

@pytest.mark.acceptance
def test_walking_skeleton():
    """This is the high-level acceptance test that gives a rough overview of how the program works.
    While I do think I have a """
    # 1. instantiation:
    model: LLM = LLM("liquid/lfm2.5-1.2b", "http://localhost:1234/v1")
    dataset_folder: Path = Path(".") / "datasets"
    remaining_attempts: int = 2

    # 2. construct the christmas data:
    data: Dataset = construct_dataset(dataset_folder = dataset_folder, topic="christmas")
    
    # 3. construct the prompt generators:
    summary_agent = Agent(model, data)
    query_agent = Agent(model, data)



    # 4. construct the forward mover:
    forward_mover = ForwardFlower(data,
                                summary_agent,
                                query_agent,
                                remaining_attempts)
    
    # 5. keep trying to construct the best summary until no more attempts, or satisfying summary:
    generated_summary, inconsistent_queries, undecided_queries = forward_mover.iteratively_generate_summary()

    assert generated_summary 
