import csv
import os
from bart_score import BARTScorer

def load_tweets(tweets_path):
    tweets = []
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tweets.append(line)
    return '\n'.join(tweets)

def bart(input_path, summary, subject, result_path):
    tweets = load_tweets(input_path)

    scorer = BARTScorer(device='cpu', checkpoint='facebook/bart-large-cnn')

    tweet_list = [t for t in tweets.split('\n') if t.strip()]
    chunks, current, current_len = [], [], 0
    for t in tweet_list:
        n = len(scorer.tokenizer.tokenize(t))
        if current and current_len + n > 900:
            chunks.append('\n'.join(current))
            current, current_len = [], 0
        current.append(t)
        current_len += n
    if current:
        chunks.append('\n'.join(current))

    scores = scorer.score(chunks, [summary] * len(chunks), batch_size=1)
    score = sum(scores) / len(scores)

    rows = [
        {'metric': 'bartscore', 'group': '',
         'value': round(float(score), 4), 'significance': ''},
        {'metric': 'bartscore_chunks', 'group': '',
         'value': len(chunks), 'significance': ''},
    ]
    out_path = f'{result_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)