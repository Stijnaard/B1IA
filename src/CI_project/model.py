"""This file contains all the code related to the LLMs and everything
it interfaces with (including the tools)"""

class LLM:
    """This is the main class for the LLM work in.

    It relies on a connection to an LLM inference server running the OpenAI protocol,
    hence the model_name- and url-parameters in the constructor.
    While the base is simple, this one strips away any unnessary complexity by
    abstracting away irrelevant attributes and methods.

    While not present yet, a method will be added for adding tools
    """
    def __init__(self, model_name: str, url: str) -> None:
        pass
