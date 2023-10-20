import argparse
import os
import re
from json import JSONDecodeError

import nltk.corpus.reader.wordnet
import requests
from tqdm import tqdm
import pandas as pd
from nltk.corpus import wordnet as wn


def get_wiki_description(term):
    api = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'prop': 'description|extracts',
        'generator': 'search',
        'exintro': '',
        'exsentences': 1,
        'gsrlimit': 1,
        'gsrsearch': term
    }
    try:
        resp = requests.get(api, params=params).json()
    except JSONDecodeError:
        with open(descriptions_path, 'a+', encoding='utf-8') as f:
            f.writelines(descriptions)
        print(idx, line)
        exit()
    description = term
    no_desc = False

    try:
        page = list(resp['query']['pages'].values())[0]
    except KeyError:
        description += (' is ' + term)
        print(description)
        descriptions.append(description + '\n')

    try:
        desc = page['description']
        if page['title'] in term.lower() and len(page['title']) < len(term):
            description += ' involves '
        else:
            description += ' is '
        desc = desc[0].lower() + desc[1:]
        description += (desc + '.')
    except KeyError:
        no_desc = True
    try:
        extract = page['extract']
        if no_desc:
            description += ':'
        clear = re.compile('\\n|<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
        extract = re.sub(clear, '', extract)
        description += (' ' + extract)
    except KeyError:
        if no_desc:
            description += (' is ' + term)
    description.replace('\t', ' ')
    return description


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--dataset', default=None, type=str, help='dataset to generate description')
    args.add_argument('-m', '--mode', default='wikipedia', type=str, help='use wikipedia or wordnet')
    args.add_argument('-v', '--verb', default=False, type=bool, help='whether the term is verb')
    args.add_argument('-b', '--batch', default=10, type=bool, help='batch for wikipedia api')
    args = args.parse_args()
    assert args.mode in ['wikipedia', 'wordnet'], 'mode must in wikipedia and wordnet'

    path = args.dataset
    dataset_filename = ''
    for file in os.listdir(path):
        if file.endswith('.terms'):
            dataset_filename = file.split('.')[0]
    assert len(dataset_filename), 'cannot find term file'
    terms_path = os.path.join(path, dataset_filename + '.terms')
    descriptions_path = os.path.join(path, dataset_filename + '.desc')

    with open(terms_path, 'r') as f:
        lines = f.read().splitlines()

    if args.mode == 'wikipedia':
        print('Wikipedia mode.')

        batch, descriptions = list(), list()
        for idx, line in enumerate(tqdm(lines)):
            term = line.split('\t')[1]
            description = get_wiki_description(term)
            print(description)
            descriptions.append(term + '\t' + description + '\n')

    else:  # args.mode == 'wordnet'
        print('WordNet mode.')
        train_data = pd.read_csv(os.path.join('SemEval', 'training.data.tsv'), index_col=2, sep='\t', header=None)
        val_data = pd.read_csv(os.path.join('SemEval', 'trial.data.tsv'), index_col=2, sep='\t', header=None)
        test_data = pd.read_csv(os.path.join('SemEval', 'test.data.tsv'), index_col=2, sep='\t', header=None)
        descriptions = list()
        for idx, line in enumerate(tqdm(lines)):
            synset, term_name = line.split('\t')
            try:
                link = ' is to ' if args.verb else ' is '
                description = synset.split('.')[0] + link + wn.synset(synset).definition()
            except nltk.corpus.reader.wordnet.WordNetError:
                link = ' is '
                name, term = term_name.split('||')
                if term.startswith('train'):
                    def_idx = term[6:]
                    definition = train_data.loc[def_idx][3]
                elif term.startswith('val'):
                    def_idx = term[4:]
                    definition = val_data.loc[def_idx][3]
                else:
                    def_idx = term[5:]
                    definition = test_data.loc[def_idx][3]
                definition = definition[0].lower() + definition[1:]
                description = name + link + definition
            description = description.replace('_', ' ')
            description = description.replace('\t', ' ')
            print(description)
            descriptions.append(term_name.replace('_', ' ') + '\t' + description + '\n')

    with open(descriptions_path, 'a+', encoding='utf-8') as f:
        f.writelines(descriptions)
