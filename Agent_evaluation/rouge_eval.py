import csv
import os
from rouge import Rouge

def rou(summary, annotator_path, subject, result_path):
    AI_summary = summary

    with open(annotator_path, 'rb') as file:
        raw = file.read()
    try:
        Annotator_summary = raw.decode('utf-8')
    except UnicodeDecodeError:
        Annotator_summary = raw.decode('mac_roman')
    Annotator_summary = Annotator_summary.replace('\r', ' ').strip()

    rouge = Rouge()
    scores = rouge.get_scores(AI_summary, Annotator_summary)[0]

    rows = []
    for variant, values in scores.items():
        for measure in ['p', 'r', 'f']:
            rows.append({'metric': variant.replace('-', '_'), 'group': measure,
                         'value': round(float(values[measure]), 4), 'significance': ''})

    out_path = f'{result_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)