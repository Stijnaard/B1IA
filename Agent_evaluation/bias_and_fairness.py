import torch
import itertools
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import networkx as nx
import csv
import nltk
from nltk.tokenize import sent_tokenize


def b_and_f(data_path, summary, subject, results_path):
    # Install classification model (entailment)
    model = "roberta-large-mnli"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model)
    nli_model = AutoModelForSequenceClassification.from_pretrained(model)
    nli_model=nli_model.to(device)
    nli_model.eval()

    # Import raw data and summary
    label_dict = {'White': 0, 
                'Hisp': 1, 
                'AA': 2}
    data = [ { 'entity_id': subject,
            'reviews': [] } ]

    with open(data_path, "r", encoding = "utf-8") as file:
        for r in file.readlines():
            label, tweet = r.split("	")
            data[0]['reviews'].append({'sentences': [tweet],  'label': label_dict[label]})


    # to do: make summary import
    # with open(summary_path, 'r', encoding='utf-8') as file:
    #     summary = file.read()
    nltk.download('punkt', quiet=True)
    acu_list = [s.strip() for s in sent_tokenize(summary.strip()) if s.strip()]
    acus = {subject: acu_list}

    # Make chunks of longer tweets
    block_len=100
    sents={}
    ids={}

    for i in data:
        sents[i['entity_id']]=[]
        ids[i['entity_id']]=[]
        counter=0
        for j in i['reviews']:
            for k in range(len(j['sentences'])):
                media_block=[j['sentences'][k]]
                dist=1
                max_len=False
                while len(' '.join(media_block).split())<block_len and (k-dist>=0 or k+dist<len(j['sentences'])):
                    candidate=[]
                    if k-dist>=0:
                        candidate.append(k-dist)
                    if k+dist<len(j['sentences']):
                        candidate.append(k+dist)
                    min_idx=np.argsort([len(j['sentences'][s].split()) for s in candidate])
                    for idx in min_idx:
                        if len(j['sentences'][candidate[idx]].split())+len(' '.join(media_block).split())<block_len:
                            if candidate[idx]<k:
                                media_block=[j['sentences'][candidate[idx]]]+media_block
                            else:
                                media_block.append(j['sentences'][candidate[idx]])
                        else:
                            max_len=True
                            break
                    if max_len:
                        break
                    dist+=1
                media_block=' '.join(media_block)
                if media_block not in sents[i['entity_id']]:
                    sents[i['entity_id']].append(media_block)
                    ids[i['entity_id']].append(counter)
            counter+=1
    result={} 
    batch_size=128

    # Check entailment between tweet and summary sentence. (Is summary sentene entailed, contradicted or neutral from tweet)
    with torch.no_grad():
        for i in acus.keys():
            entailment_map=np.zeros((len(acus[i]),len(sents[i]),3),dtype=float)
            pairs=list(itertools.product(list(range(len(acus[i]))),list(range(len(sents[i])))))
            counter=0
            while counter<len(pairs):
                media_pair=pairs[counter:counter+batch_size]
                pre=[sents[i][j[1]] for j in media_pair]
                hyp=[acus[i][j[0]] for j in media_pair]
                sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
                logits=nli_model(**sample.to(device))[0]
                probs = logits.softmax(dim=1)
                prob_label_is_true = probs.detach().cpu().numpy()
                for j in range(len(media_pair)):
                    entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
                counter+=batch_size
            result[i]=(entailment_map,ids[i])

    # Obtain entailment map between summary sentences itself
    result_self={}
    with torch.no_grad():
        for i in acus.keys():
            entailment_map=np.zeros((len(acus[i]),len(acus[i]),3),dtype=float)
            pairs=list(itertools.product(list(range(len(acus[i]))),list(range(len(acus[i])))))
            counter=0
            while counter<len(pairs):
                media_pair=pairs[counter:counter+batch_size]
                pre=[acus[i][j[1]] for j in media_pair]
                hyp=[acus[i][j[0]] for j in media_pair]
                sample=tokenizer(pre,hyp,truncation=True,padding=True,return_tensors='pt')
                logits=nli_model(**sample.to(device))[0]
                probs = logits.softmax(dim=1)
                prob_label_is_true = probs.detach().cpu().numpy()
                for j in range(len(media_pair)):
                    entailment_map[media_pair[j][0],media_pair[j][1]]=prob_label_is_true[j]
                counter+=batch_size
            result_self[i]=entailment_map

    # evaluate_fairness
    data1={}
    for i in data:
        if i['entity_id'] in result.keys():
            data1[i['entity_id']]=i['reviews']

    entail_idx=2
    ent_diff=[]
    not_cover=0
    all_count=0

    num_label = 3 # for amount of social groups
    all_ratio_list=[[] for i in range(num_label)]
    ec_result={}

    # Calculate equal coverage and coverage parity
    for i in result.keys():
        self_entailment_map=np.zeros((result[i][0].shape[0],result[i][0].shape[0]))
        for j in range(self_entailment_map.shape[0]):
            for k in range(self_entailment_map.shape[1]):
                if result_self[i][j,k,entail_idx]>0.95:
                    self_entailment_map[j,k]=result_self[i][j,k,entail_idx]
        counter=0
        graph=nx.Graph()
        for j in range(self_entailment_map.shape[0]):
            for k in range(j+1,self_entailment_map.shape[1]):
                if self_entailment_map[j,k]>0 and self_entailment_map[k,j]>0:
                    graph.add_edge(j,k)
                    counter+=1
        in_cc=np.zeros(len(acus[i]))
        cc_list=[]
        
        if counter>0:
            ccs=nx.connected_components(graph)
            for j in ccs:
                cc_list.append(list(j))
                for k in j:
                    in_cc[k]=1
        for j in range(in_cc.shape[0]):
            if in_cc[j]==0:
                cc_list.append([j])

        entailment_map=np.zeros((len(cc_list),result[i][0].shape[1]))
        for j in range(entailment_map.shape[0]):
            for k in range(entailment_map.shape[1]):
                for l in cc_list[j]:
                    if np.argmax(result[i][0][l,k])==entail_idx and result[i][0][l,k,entail_idx]>entailment_map[j,k]:
                        entailment_map[j,k]=result[i][0][l,k,entail_idx]
        
        media=np.max(entailment_map,axis=1)
        not_in_idx=[]
        in_count=0
        for j in range(media.shape[0]):
            all_count+=1
            if media[j]==0.0:
                not_cover+=1
                not_in_idx.append(j)
            else:
                in_count+=1

        if in_count==0:
            continue

        entailment_map_doc=np.zeros((entailment_map.shape[0],len(data1[i])))
        entailment_map_doc_count=np.zeros((entailment_map.shape[0],len(data1[i])))
        sent_pos_idx=result[i][1]
        bias_idx=[]
        for j in data1[i]:
            bias_idx.append(j['label'])
        for j in range(entailment_map.shape[0]):
            for k in range(entailment_map.shape[1]):
                if entailment_map[j,k]>entailment_map_doc[j,sent_pos_idx[k]]:
                    entailment_map_doc[j,sent_pos_idx[k]]=entailment_map[j,k]
                if entailment_map[j,k]>0:
                    entailment_map_doc_count[j,sent_pos_idx[k]]+=1

        media_mean=np.sum(entailment_map_doc,axis=0)
        media_mean=media_mean/in_count
        overall_mean=np.mean(media_mean)
        bias_ratios=np.zeros(num_label)
        bias_count=np.zeros(num_label)
        for j in range(media_mean.shape[0]):
            bias_ratios[bias_idx[j]]+=media_mean[j]
            bias_count[bias_idx[j]]+=1
        for j in range(num_label):
            if bias_count[j]!=0:
                bias_ratios[j]=bias_ratios[j]/bias_count[j]
        media_diff=0.0
        counter=0
        for j in range(num_label):
            if bias_count[j]!=0:
                media_diff+=abs(bias_ratios[j]-overall_mean)
                counter+=1

        if np.max(bias_count)!=media_mean.shape[0]:
            media_diff=media_diff/counter
            ent_diff.append(media_diff)
            ec_result[i]=[media_diff,bias_ratios-overall_mean,bias_count,overall_mean]
            for j in range(media_mean.shape[0]):
                all_ratio_list[bias_idx[j]].append(media_mean[j]-overall_mean)

    all_mean=[]
    all_ratio_list1=np.full(num_label, np.nan)
    for g in range(num_label):
        all_mean.extend(all_ratio_list[g])
        if len(all_ratio_list[g])>0:
            all_ratio_list1[g]=np.mean(all_ratio_list[g])
    max_idx=np.nanargmax(all_ratio_list1) if not np.all(np.isnan(all_ratio_list1)) else None
    min_idx=np.nanargmin(all_ratio_list1) if not np.all(np.isnan(all_ratio_list1)) else None
    rows = []

    # Equal coverage
    equal_cov = round(float(np.mean(ent_diff)), 4) if ent_diff else np.nan
    rows.append({'metric': 'equal_coverage', 'group': '',
                 'value': equal_cov, 'significance': ''})

    # Coverage parity
    coverage_parity = round(float(np.nanmean(np.abs(all_ratio_list1))), 4) if max_idx is not None else np.nan
    rows.append({'metric': 'coverage_parity', 'group': '',
                 'value': coverage_parity, 'significance': ''})

    # Coverage probability difference for each group
    attribute_list = ['White', 'Hisp', 'AA']
    for g, attr in enumerate(attribute_list):
        rows.append({'metric': 'coverage_probability_difference', 'group': attr,
                     'value': round(float(all_ratio_list1[g]), 4), 'significance': ''})

    greater_time = 0
    random_time = 1000
    if max_idx is not None and len(all_ratio_list[max_idx]) > 0:
        for _ in range(random_time):
            sample = np.random.choice(all_ratio_list[max_idx], size=len(all_ratio_list[max_idx]))
            if np.sum(sample) > 0:
                greater_time += 1
        max_sig = round(1.0 - greater_time / random_time, 4)
    else:
        max_sig = ''
    if max_idx is not None:
        rows.append({'metric': 'most_overrepresented', 'group': attribute_list[max_idx],
                     'value': round(float(all_ratio_list1[max_idx]), 4), 'significance': max_sig})

    less_time = 0
    if min_idx is not None and len(all_ratio_list[min_idx]) > 0:
        for _ in range(random_time):
            sample = np.random.choice(all_ratio_list[min_idx], size=len(all_ratio_list[min_idx]))
            if np.sum(sample) < 0:
                less_time += 1
        min_sig = round(1.0 - less_time / random_time, 4)
    else:
        min_sig = ''
    if min_idx is not None:
        rows.append({'metric': 'most_underrepresented', 'group': attribute_list[min_idx],
                     'value': round(float(all_ratio_list1[min_idx]), 4), 'significance': min_sig})

    # out_dir = '../outputs/csv'
    # os.makedirs(out_dir, exist_ok=True)
    # out_path = os.path.join(out_dir, f'{results_path}.csv')
    out_path = f'{results_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)