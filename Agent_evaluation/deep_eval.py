import csv
import os
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel

def load_tweets(tweets_path):
    tweets = []
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tweets.append(line)
    return '\n'.join(tweets)

def deep_ev(input_path, summary, subject, result_path):
    tweets = load_tweets(input_path)

    model = OllamaModel(model="qwen3.5:9b")
    metric = SummarizationMetric(threshold=0.5, model=model)
    test_case = LLMTestCase(input=tweets, actual_output=summary)

    metric.measure(test_case)

    score = round(float(metric.score), 4) if metric.score is not None else ''
    rows = [
        {'metric': 'summarization_score', 'group': '',
         'value': score, 'significance': ''},
        {'metric': 'summarization_success', 'group': '',
         'value': str(metric.success), 'significance': ''},
        {'metric': 'summarization_reason', 'group': '',
         'value': metric.reason, 'significance': ''},
    ]


    out_path = f'{result_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)