from bias_and_fairness import b_and_f 
from neutralisation import neut
from ratio import rat
from deep_eval import deep_ev
from bart_score_eval import bart
from rouge_eval import rou

import json


def main():
    topics = ['Christmas', 'Netflix', 'McDonalds']

    for t in topics:
        input_path = f'../inputs/{t}.txt'
        clean_input_path = f'../inputs/clean/{t}.txt'
        for i in ['1', '2', 'baseline']:
            name = f'{t}_{i}'
            summary_path = f'../outputs/{name}.json'
            with open(summary_path, 'r', encoding='utf-8') as file:
                output = json.load(file)

            summary = output['state']['summary']

            # bias and fairness (equal coverage and coverage parity)
            print('---- Calculate bias and fairness ----')
            result_path = f'../results/{name}_{i}_bias_and_fairness'
            b_and_f(input_path, summary, t, result_path)

            # neutralisation
            print('---- Calculate neutralisation ----')
            result_path = f'../results/{name}_{i}_neutralisation'
            neut(clean_input_path, summary, t, result_path)

            # ratio
            print('---- Calculate ratio ----')
            result_path = f'../results/{name}_{i}_ratio'
            rat(clean_input_path, summary, t, result_path)

            # deep evaluation
            print('---- Calculate deep eval ----')
            try:
                result_path = f'../results/{name}_{i}_deep_eval'
                deep_ev(clean_input_path, summary, t, result_path)
            except: 
                print('Deep evaluation does not work')            

            # BARTscore
            print('---- Calculate BART score ----')
            result_path = f'../results/{name}_{i}_BART'
            bart(clean_input_path, summary, t, result_path)


            # ROUGE
            print('---- Calculate ROUGE ----')
            for j in [1,2]:
                result_path = f'../results/{name}_{i}_annotator_{j}_ROUGE'
                annotator_path = f'../Annotations/{t}_Annotator{j}.txt'
                rou(summary, annotator_path, t, result_path)








if __name__ == "__main__":
    main()
