from CI_project.model import LLM
from CI_project.dataset import Dataset
from CI_project.prompt import PromptGenerator

from typing import Optional, Iterable
class ForwardFlower:
    def __init__(self, model: LLM, 
                 dataset: Dataset,
                 summary_prompt_generator: PromptGenerator,
                 revision_prompt_generator: PromptGenerator):

                 self.model = model
                 self.dataset = dataset
                 self.summary_prompt_generator = summary_prompt_generator
                 self.revision_prompt_generator = revision_prompt_generator

    def forward(self, previous_summary: Optional[str] = None, previous_inconsistent_queries: Optional[Iterable[str]] = None) -> tuple[str, Iterable[str], Iterable[str]]:
        pass

