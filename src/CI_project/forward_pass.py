from CI_project.model import LLM
from CI_project.dataset import Dataset
from CI_project.prompt import PromptGenerator

from typing import Iterable
class ForwardFlower:
    def __init__(self, model: LLM, 
                 dataset: Dataset,
                 summary_prompt_generator: PromptGenerator,
                 revision_prompt_generator: PromptGenerator,
                 n_attempts: int):

                 self.model = model
                 self.dataset = dataset
                 self.summary_prompt_generator = summary_prompt_generator
                 self.revision_prompt_generator = revision_prompt_generator
                 self.n_attempts: int = n_attempts

    def iteratively_generate_summary(self) -> tuple[str, Iterable[str], Iterable[str]]:
        return "fake summary #1", ['inconsistent query 1', 'inconsistent query 2'], []

