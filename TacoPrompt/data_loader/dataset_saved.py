import os
import pickle
import random
import time
from collections import deque
from itertools import chain, product

import networkx as nx
import spacy
from networkx.algorithms import descendants
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000

nlp = spacy.load('en_core_web_md')


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0,
                 c_count=0, create_date="None", use_wordnet=True, single_word=False):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        self.use_wordnet = use_wordnet
        self.single_word = single_word
        self.description = ''
        self.nlp_description = nlp('')

    def set_description(self, description):
        self.description = description
        self.nlp_description = nlp(description)

    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)

    def __lt__(self, other):
        if self.display_name < other.display_name:
            return True
        else:
            return False

    def __eq__(self, other):
        if self.display_name == other.display_name:
            return True
        else:
            return False

    def __hash__(self):
        return hash(self.display_name)


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=False, existing_partition=True, partition_pattern='internal'):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.data_path = path

        if raw:
            self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        print('loading pickled dataset')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        self.name = data["name"]
        self.taxonomy = data['taxonomy']
        self.node_id2taxon = data['id2taxon']
        self.taxon2node_id = data['taxon2id']
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

        # path = os.path.split(pickle_path)[0]
        #
        # with open(os.path.join(path, f'{self.name}.terms.train'), 'w') as f:
        #     for id in self.train_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.validation'), 'w') as f:
        #     for id in self.validation_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.test'), 'w') as f:
        #     for id in self.test_node_ids:
        #         f.write(str(id) + '\n')
        # exit(0)
        print('dataset loaded')

    def _load_dataset_raw(self, dir_path):
        path = os.path.split(dir_path)[0]
        node_file_name = os.path.join(path, f"{self.name}.terms")
        edge_file_name = os.path.join(path, f"{self.name}.taxo")
        desc_file_name = os.path.join(path, f"{self.name}.desc")
        if self.embed_suffix == "":
            output_pickle_file_name = os.path.join(path, f"{self.name}.pickle.bin")
        else:
            output_pickle_file_name = os.path.join(path, f"{self.name}.{self.embed_suffix}.pickle.bin")

        tx_id2taxon = {}
        self.taxonomy = nx.DiGraph()

        # load nodes
        with open(desc_file_name, "r", encoding='utf-8') as fdesc:
            with open(node_file_name, "r", encoding='utf-8') as fin:
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
                            # assert len(segs_desc) == 1
                            desc = segs_desc[0]
                        taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                        taxon.set_description(desc)
                        tx_id2taxon[segs[0]] = taxon
                        self.taxonomy.add_node(taxon)
        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    self.taxonomy.add_edge(parent_taxon, child_taxon)

        # generate vocab
        # tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        self.tx_id2node_id = {node.tx_id: idx for idx, node in enumerate(self.taxonomy.nodes())}
        self.node_id2tx_id = {v: k for k, v in self.tx_id2node_id.items()}
        self.node_id2taxon = {k: tx_id2taxon[v] for k, v in self.node_id2tx_id.items()}
        self.taxon2node_id = {v: k for k, v in self.node_id2taxon.items()}
        self.vocab = [tx_id2taxon[self.node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in
                      self.node_id2tx_id]

        if self.existing_partition:
            # Use the pickled partitions
            # with open(os.path.join(path, 'split_node_ids.pickle'), 'rb') as f:
            #     data = pickle.load(f)
            # self.validation_node_ids = data["validation_node_ids"]
            # self.test_node_ids = data["test_node_ids"]
            # self.train_node_ids = data["train_node_ids"]

            # Or use the train/val/test files
            dir_path = os.path.dirname(dir_path)
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")

            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

            self.train_node_ids = [int(n) for n in raw_train_node_list]
            self.validation_node_ids = [int(n) for n in raw_validation_node_list]
            self.test_node_ids = [int(n) for n in raw_test_node_list]

        else:
            print("Partition graph ...")
            if self.partition_pattern == 'leaf':
                sampled_node_ids = []
                for node in self.taxonomy.nodes():
                    if self.taxonomy.out_degree(node) == 0:
                        sampled_node_ids.append(self.tx_id2node_id[node.tx_id])
                random.seed(47)
                random.shuffle(sampled_node_ids)
            elif self.partition_pattern == 'internal':
                root_node = [node for node in self.taxonomy.nodes() if self.taxonomy.in_degree(node) == 0]
                sampled_node_ids = [self.tx_id2node_id[node.tx_id] for node in self.taxonomy.nodes() if
                                    node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)
            else:
                raise ValueError('Unknown partition method!')

            validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = sampled_node_ids[:validation_size]
            self.test_node_ids = sampled_node_ids[validation_size:(validation_size + test_size)]
            self.train_node_ids = [node_id for node_id in self.node_id2tx_id if
                                   node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            print("Finish partitioning graph ...")

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            data = {
                "name": self.name,
                "taxonomy": self.taxonomy,
                "id2taxon": self.node_id2taxon,
                "taxon2id": self.taxon2node_id,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32, max_pos_size=100,
                 expand_factor=64, cache_refresh_time=128, test_topk=-1, tokenizer=None):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size
        self.max_pos_size = max_pos_size
        self.expand_factor = expand_factor
        self.cache_refresh_time = cache_refresh_time
        self.test_topk = test_topk
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.taxon2id = graph_dataset.taxon2node_id
        self.id2taxon = graph_dataset.node_id2taxon

        full_graph = graph_dataset.taxonomy
        train_nodes = [self.id2taxon[node_id] for node_id in graph_dataset.train_node_ids]
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        self.pseudo_root_node = Taxon(tx_id='', norm_name='', display_name='pseudo root')
        full_graph.add_node(self.pseudo_root_node)
        for node in roots:
            full_graph.add_edge(self.pseudo_root_node, node)
        train_nodes.append(self.pseudo_root_node)
        self.full_graph = full_graph

        if mode == 'train':
            # add pseudo leaf node to core graph
            datapath = os.path.split(graph_dataset.data_path)[0]
            graph_pickle_path = os.path.join(datapath, 'subgraphs.pickle')
            graph_pickled = False
            if os.path.isfile(graph_pickle_path):
                graph_pickled = True
                with open(graph_pickle_path, 'rb') as f:
                    graphs = pickle.load(f)

            print('adding pseudo leaf')
            if graph_pickled:
                self.core_subgraph = graphs['core_subgraph']
                self.pseudo_leaf_node = graphs['pseudo_leaf_node']
            else:
                self.core_subgraph = self._get_holdout_subgraph(train_nodes)
                self.pseudo_leaf_node = Taxon(tx_id='', norm_name='', display_name='pseudo leaf')
                self.core_subgraph.add_node(self.pseudo_leaf_node)
                for node in list(self.core_subgraph.nodes()):
                    self.core_subgraph.add_edge(node, self.pseudo_leaf_node)

            self.taxon2id[self.pseudo_leaf_node] = len(full_graph.nodes)
            self.taxon2id[self.pseudo_root_node] = len(full_graph.nodes) - 1
            self.id2taxon[len(full_graph.nodes)] = self.pseudo_leaf_node
            self.id2taxon[len(full_graph.nodes) - 1] = self.pseudo_root_node
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            # self.node_list = [n for n in train_nodes if n not in roots]
            self.node_list = [n for n in train_nodes if n != self.pseudo_root_node]

            # build node2pos, node2edge
            print('building node2pos, node2edge')
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children = {}, {}
            for node in self.node_list:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre != suc]
                if len(node_pos_edges) == 0:
                    node_pos_edges = [(pre, suc) for pre in parents for suc in children]

                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(
                    set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children

            print('building valid and test node list')
            self.valid_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.validation_node_ids]
            if graph_pickled:
                self.valid_holdout_subgraph = graphs['valid_subgraph']
            else:
                self.valid_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.valid_node_list)
                self.valid_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.valid_holdout_subgraph.nodes() if
                             self.valid_holdout_subgraph.out_degree(node) == 0]:
                    self.valid_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.valid_id2taxon = {idx: taxon for idx, taxon in enumerate(self.valid_holdout_subgraph.nodes())}
            self.valid_taxon2id = {v: k for k, v in self.valid_id2taxon.items()}
            self.valid_node2pos = self._find_insert_position(self.valid_node_list, self.valid_holdout_subgraph)

            self.test_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.test_node_ids]
            if graph_pickled:
                self.test_holdout_subgraph = graphs['test_subgraph']
            else:
                self.test_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.test_node_list)
                self.test_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.test_holdout_subgraph.nodes() if
                             self.test_holdout_subgraph.out_degree(node) == 0]:
                    self.test_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.test_id2taxon = {idx: taxon for idx, taxon in enumerate(self.test_holdout_subgraph.nodes())}
            self.test_taxon2id = {v: k for k, v in self.test_id2taxon.items()}
            self.test_node2pos = self._find_insert_position(self.test_node_list, self.test_holdout_subgraph)

            if not graph_pickled:
                with open(graph_pickle_path, 'wb') as f:
                    pickle.dump({
                        'pseudo_leaf_node': self.pseudo_leaf_node,
                        'core_subgraph': self.core_subgraph,
                        'valid_subgraph': self.valid_holdout_subgraph,
                        'test_subgraph': self.test_holdout_subgraph
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # used for sampling negative positions during train/validation stage
            self.pointer = 0
            self.all_edges = list(self._get_candidate_positions(self.core_subgraph))
            random.shuffle(self.all_edges)

        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        """ Generate an data instance based on train/validation/test mode.

        If self.sampling_mode == 0:
            This list may contain more than one triplets with label = 1
        If self.sampling_mode == 1:
            This list contain one and ONLY one triplet with label = 1, others have label = 0
        """
        res = []
        q = self.node_list[idx]

        # generate positive triplet(s)
        if self.sampling_mode == 0:
            positive_positions = self.node2pos[q]
            if len(positive_positions) > self.max_pos_size and self.mode == 'train':
                positive_positions = random.sample(positive_positions, k=self.max_pos_size)
        else:  # self.sampling_mode > 0
            positive_positions = [random.choice(self.node2pos[q])]

        # select negative triplet(s)
        negative_size = len(res) if self.negative_size == -1 else self.negative_size
        negative_positions = self._get_negative_positions(q, negative_size)

        # Tags: position, parent, child, best, worst
        for p, c in positive_positions:
            best, worst = self.get_best_worst_siblings(p, q, g=self.core_subgraph)
            res.append([q.description, p.description, c.description, best.description, worst.description, 1, 1, 1, 1, 1,
                        bool(p.description),
                        bool(c.description),
                        bool(best.description),
                        bool(worst.description)])
        for p, c in negative_positions:
            best, worst = self.get_best_worst_siblings(p, q, g=self.core_subgraph)
            res.append([q.description, p.description, c.description, best.description, worst.description, 0,
                        int(p in self.node2parents[q]),
                        int(c in self.node2children[q]),
                        int(not set(self.core_subgraph.predecessors(best)).isdisjoint(
                            set(self.core_subgraph.predecessors(q)))),
                        int(not set(self.core_subgraph.predecessors(worst)).isdisjoint(
                            set(self.core_subgraph.predecessors(q)))),
                        bool(p.description),
                        bool(c.description),
                        bool(best.description),
                        bool(worst.description)])
            # if res[-1][8] == 1 and bool(best.description):
            #     print(q.description, best.description)
            #     print(list(set(self.core_subgraph.predecessors(q)).intersection(set(self.core_subgraph.predecessors(best))))[0].display_name)

        return tuple(res)

    def get_best_worst_siblings(self, parent, query, g=None, train=True):
        probability = 0.9
        best_score, worst_score = -2, 2
        best, worst = self.pseudo_leaf_node, self.pseudo_leaf_node
        sibling_list = list(g.successors(parent))
        # if parent in g.predecessors(query):
        #     sibling_list += list(g.successors(query))
        no_pseudo = False
        if len(sibling_list) > 1:
            no_pseudo = True
        for n in sibling_list:
            if n == query or (n.description == '' and no_pseudo):
                continue
            semantic_similarity = query.nlp_description.similarity(n.nlp_description)
            if semantic_similarity > best_score:
                best = n
                best_score = semantic_similarity
            if semantic_similarity < worst_score:
                worst = n
                worst_score = semantic_similarity
        r = random.choices([True, False], [probability, 1 - probability])[0]
        if train:
            if r:
                best = random.sample(sibling_list, k=1)[0]
            r = random.choices([True, False], [probability, 1 - probability])[0]
            if r:
                worst = random.sample(sibling_list, k=1)[0]
        return best, worst

    def _get_holdout_subgraph(self, nodes):
        node_to_remove = [n for n in self.full_graph.nodes if n not in nodes]
        subgraph = self.full_graph.subgraph([node for node in nodes]).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        if subgraph.in_degree(s) > 1:
                            subgraph.remove_edge(node, s)
        return subgraph

    def _get_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates

    def _find_insert_position(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
            parents = set()
            children = set()
            ps = deque(holdout_graph.predecessors(node))
            cs = deque(holdout_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(holdout_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(holdout_graph.successors(c))
            if not children:
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p != c]
            node2pos[node] = position
        return node2pos

    def _get_negative_positions(self, query_node, negative_size):
        if self.sampling_mode == 0:
            return self._get_at_most_k_negatives(query_node, negative_size)
        elif self.sampling_mode == 1:
            return self._get_exactly_k_negatives(query_node, negative_size)

    def _get_at_most_k_negatives(self, query_node, negative_size):
        """ Generate AT MOST negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        while True:
            negatives = [ele for ele in self.all_edges[self.pointer: self.pointer + negative_size] if
                         ele not in self.node2pos[query_node] and ele not in self.node2edge[query_node]]
            if len(negatives) > 0:
                break
            self.pointer += negative_size
            if self.pointer >= len(self.all_edges):
                self.pointer = 0

        return negatives

    def _get_exactly_k_negatives(self, query_node, negative_size, ignore=[]):
        """ Generate EXACTLY negative_size samples for the query node
        """
        if self.pointer == 0:
            random.shuffle(self.all_edges)

        negatives = []
        while len(negatives) != negative_size:
            n_lack = negative_size - len(negatives)
            negatives.extend([ele for ele in self.all_edges[self.pointer: self.pointer + n_lack] if
                              ele not in self.node2pos[query_node] and ele not in self.node2edge[
                                  query_node] and ele not in ignore])
            self.pointer += n_lack
            if self.pointer >= len(self.all_edges):
                self.pointer = 0
                random.shuffle(self.all_edges)
        if len(negatives) > negative_size:
            negatives = negatives[:negative_size]

        return negatives
