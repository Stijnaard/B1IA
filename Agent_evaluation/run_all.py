from bias_and_fairness import b_and_f 
from neutralisation import neut
from ratio import rat
from deep_eval import deep_ev
from bart_score_eval import bart
from rouge_eval import rou

import json

def get_results(input_path, clean_input_path, summary, name, t):
    # bias and fairness (equal coverage and coverage parity)
    print('---- Calculate bias and fairness ----')
    result_path = f'../results/raw/{name}_bias_and_fairness'
    b_and_f(input_path, summary, t, result_path)

    # neutralisation
    print('---- Calculate neutralisation ----')
    result_path = f'../results/raw/{name}_neutralisation'
    neut(clean_input_path, summary, t, result_path)

    # ratio
    print('---- Calculate ratio ----')
    result_path = f'../results/raw/{name}_ratio'
    rat(clean_input_path, summary, t, result_path)

    # deep evaluation
    print('---- Calculate deep eval ----')
    try:
        result_path = f'../results/raw/{name}_deep_eval'
        deep_ev(clean_input_path, summary, t, result_path)
    except: 
        print('Deep evaluation does not work')            

    # BARTscore
    print('---- Calculate BART score ----')
    result_path = f'../results/raw/{name}_BART'
    bart(clean_input_path, summary, t, result_path)


    # ROUGE
    print('---- Calculate ROUGE ----')
    for j in [1,2]:
        result_path = f'../results/raw/{name}_annotator_{j}_ROUGE'
        annotator_path = f'../Annotations/{t}_Annotator{j}.txt'
        rou(summary, annotator_path, t, result_path)

import os
import pandas as pd

def summarize_results(name, m, versions):
    dfs = []
    for v in versions:
        path = f'../results/raw/{name}_{v}_{m}.csv'
        if os.path.exists(path):          # deep_eval kan ontbreken
            dfs.append(pd.read_csv(path))
    if not dfs:
        return

    df = pd.concat(dfs)
    keys = [c for c in df.columns if c not in ('value', 'significance')]

    out = (df.groupby(keys, sort=False, dropna=False)['value']
             .agg(['mean', 'std'])
             .reset_index())

    os.makedirs('../results/agg', exist_ok=True)
    out.to_csv(f'../results/agg/{name}_{m}.csv', index=False)


def main():
    topics = ['Christmas', 'Netflix', 'McDonalds']
    user_prompt = [1, 2]
    alt_v = ['a1_only', 'a2_only', 'baseline', 'full']
    version = ['v0', 'v1', 'v2']
    metrics = ['bias_and_fairness', 'neutralisation', 'ratio', 'deep_eval', 'BART',
            'annotator_1_ROUGE', 'annotator_2_ROUGE']

    for t in topics:
        input_path = f'../inputs/{t}.txt'
        clean_input_path = f'../inputs/clean/{t}.txt'

        for up in user_prompt:
            for av in alt_v:
                for v in version:
                    base = f'{t}_up-{up}_av-{av}_{v}'
                    with open(f'../Agent/logs/{base}.json', 'r', encoding='utf-8') as file:
                        summary = json.load(file)
                    summary = summary['state']['summary']
                    get_results(input_path, clean_input_path, summary, base, t)
                for m in metrics:
                    name = f'{t}_up-{up}_av-{av}'
                    summarize_results(name, m, version)
                    
if __name__ == "__main__":
    main()
