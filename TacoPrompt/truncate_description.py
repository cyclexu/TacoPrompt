import os
import pickle
import random
import time
from collections import deque
from itertools import chain, product

import networkx as nx
from networkx.algorithms import descendants
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import torch
import numpy as np
import argparse

MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000
TRUNCATE_LENGTH = 80

def truncate(desc_file_name, node_file_name, out_file_name, tokenizer):
    with open(desc_file_name, "r", encoding='utf-8') as fdesc:
        with open(node_file_name, "r", encoding='utf-8') as fin:
            with open(out_file_name, 'w', encoding='utf-8') as fout:
                for line, desc in tqdm(zip(fin, fdesc), desc="Loading terms"):
                    line = line.strip()
                    desc = desc.strip()
                    if line:
                        segs = line.split("\t")
                        segs_desc = desc.split("\t")
                        assert len(segs) == 2, f"Wrong number of segmentations {line}"
                        try:
                            assert segs[1] == segs_desc[0]
                            desc = segs_desc[1]
                        except AssertionError:
                            desc = segs_desc[0]
                    token_ids = tokenizer(desc, padding=False, return_tensors="pt",add_special_tokens=False)
                    # limit the length of the description to TRUNCATE_LENGTH
                    desc_new = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(token_ids["input_ids"][0,:TRUNCATE_LENGTH]))
                    if desc_new[-1] == '.':
                        new_line = segs[1] + '\t' + desc_new + '\n'
                    else:
                        new_line = segs[1] + '\t' + desc_new + '.' + '\n'
                    fout.write(new_line)
    
if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Truncation')
    args.add_argument('-d', '--taxo_dir', default=None, type=str, help='directory of taxonomy  (default: None)')
    args.add_argument('-n', '--taxo_name', default=None, type=str, help='name of taxonomy (default: None)')
    args = args.parse_args()
    
    taxo_dir = args.taxo_dir
    taxo_name = args.taxo_name
    
    desc_file_name = "./data/{}/{}.desc".format(taxo_dir, taxo_name)
    node_file_name = "./data/{}/{}.terms".format(taxo_dir, taxo_name)
    out_file_name = "./data/{}/{}.newdesc".format(taxo_dir, taxo_name)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    truncate(desc_file_name, node_file_name, out_file_name, tokenizer)
    
    
    
                