class Agent:
    def __init__(self, model_path: str, system_prompt: str):
        self.system_prompt: str = system_prompt
        pass

    def prompt(self, user_prompt: str) -> str:
        pass

class SummarisationAgent(Agent):
    def prompt(self, summary: str) -> list[str]:
        return ['a', 'b']