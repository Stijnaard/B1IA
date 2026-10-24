from CI_project.model import LLM
from CI_project.dataset import Dataset
from CI_project.agent import Agent
from CI_project.ontology import OntologyChecker

from typing import Iterable
class ForwardFlower:
    def __init__(self,
                 dataset: Dataset,
                 summary_agent: Agent,
                 query_agent: Agent,
                 ontology_checker: OntologyChecker,
                 n_attempts: int):

                 self.dataset = dataset
                 self.summary_agent = summary_agent
                 self.query_agent = query_agent
                 self.ontology_checker = ontology_checker
                 self.n_attempts: int = n_attempts

    def iteratively_generate_summary(self) -> tuple[str, Iterable[str], Iterable[str]]:

        all_untestable_queries: list[str] = []
        summary: str = ""

        # 1. generate summary:
        summary: str = self.summary_agent.generate_summary()

        for attempt in range(1, self.n_attempts+1):
            
            # 2. extract the queries/facts from the summary:
            queries: Iterable[str] = self.query_agent.generate_queries(summary)
            
            # 3. test them against the ontology
            accepted_queries, conflicting_queries, untestable_queries = self.ontology_checker.check(queries)
            all_untestable_queries.extend(untestable_queries)
            
            # 4. either return or revise the summary
            if not conflicting_queries:
                break
            else:
                summary = self.summary_agent.revise_summary(conflicting_queries, summary)

        return summary, conflicting_queries, all_untestable_queries
