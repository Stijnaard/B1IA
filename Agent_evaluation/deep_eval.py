from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from deepeval.models import OllamaModel


with open('../inputs/Netflix_clean.txt', 'r', encoding='utf-8') as file:
    tweets = file.read()

with open('netflix_testsum.txt', 'r', encoding='utf-8') as file:
    summary = file.read()

# tweets = '\n'.join(tweets.split('\n'))

model = OllamaModel(model="llama3.2")
metric = SummarizationMetric(threshold=0.5, model=model)

test_case = LLMTestCase(input=tweets, actual_output=summary)

# metric.measure(test_case)
# print(metric.score, metric.reason)

evaluate(test_cases=[test_case], metrics=[metric])