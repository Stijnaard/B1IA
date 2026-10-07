"""This file depicts what I want the program to look like on a high level.
As of now, this is still a rough prototype, so don't take what I currently have too seriously"""

from ontology_checker import query_ontology
from agent import Agent, SummarisationAgent
from prompts import (get_query_generation_user_prompt,
                     get_query_generation_system_prompt,
                     get_summarisation_system_prompt,
                     get_summarisation_user_prompt)

agent1 = Agent()
agent2 = SummarisationAgent()

summarisation_user_prompt = get_summarisation_user_prompt()
query_generationn_user_prompt = ...
while True:
    summary: str = agent1.prompt(summarisation_user_prompt)

    queries: list[str] = agent2.prompt(summary)

    inconsistent_queries, unprocessable_queries = query_ontology(queries)

    if not inconsistent_queries and not unprocessable_queries:
        break