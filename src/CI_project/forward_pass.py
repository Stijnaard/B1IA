from CI_project.model import LLM
from CI_project.dataset import Dataset
from CI_project.agent import Agent

from typing import Iterable
class ForwardFlower:
    def __init__(self,
                 dataset: Dataset,
                 summary_agent: Agent,
                 query_agent: Agent,
                 n_attempts: int):

                 self.dataset = dataset
                 self.summary_agent = summary_agent
                 self.query_agent = query_agent
                 self.n_attempts: int = n_attempts

    def iteratively_generate_summary(self) -> tuple[str, Iterable[str], Iterable[str]]:
        return "fake summary #1", ['inconsistent query 1', 'inconsistent query 2'], []

