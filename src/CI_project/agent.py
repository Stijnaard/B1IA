from CI_project.model import LLM

class Agent:
    def __init__(self, model: LLM):
        self.model = model