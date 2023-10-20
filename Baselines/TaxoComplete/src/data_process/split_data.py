from collections import defaultdict, deque
from model.sbert.readers import InputExample
import data_process.helpers as helpers
from itertools import chain
import networkx as nx
from mpmath import *
import random
import pdb
mp.dps = 25; mp.pretty = True
from networkx.algorithms import descendants
import pickle as pkl

class Dataset():
    def __init__(self, graph_dataset,sampling_method,neg_number,seed):
        helpers.set_seed(seed)
        full_graph = graph_dataset.taxonomy
        train_node_ids = graph_dataset.train_node_ids
        # with open('train_node_ids_tri.pkl','wb') as f:
        #     pkl.dump(train_node_ids,f)
        # print(sorted(train_node_ids))
        # exit()
        # print(len(train_node_ids))
        # print(len(full_graph.edges))
        # exit()
        roots = graph_dataset.root
        # print(len(roots))
        if len(roots) > 1:
            self.root = max(full_graph.nodes) + 1
            for r in roots:
                full_graph.add_edge(self.root, r)
            train_node_ids.append(self.root)
        else:
            self.root = roots[0]

        
        self.definitions = graph_dataset.term2def
        self.definitions[self.root] = {"label":" ","summary":" "}
        self.full_graph = full_graph
        # try:
        #     cycles = nx.find_cycle(self.full_graph, orientation="original")
        #     for tupl in cycles:
        #         self.full_graph.add_edge(self.root, tupl[0])
        # except:
        #     print("no cycles found")
            # exit()
        
        
        
        # print(len(self.full_graph.nodes))
        
        # exit()
        # print(len(self.full_graph.edges))
        
        # flag = False
        # for edge in self.full_graph.edges:
        #     if edge not in test_edges:
        #         flag = True
        #         print(edge)
        # print(flag)
        # exit()
        
        self.core_subgraph = self._get_holdout_subgraph(train_node_ids)
        print(len(self.core_subgraph.nodes))
        print(len(self.core_subgraph.edges))
        # exit()
        self.pseudo_leaf_node = max(full_graph.nodes) + 1
        self.definitions[self.pseudo_leaf_node] = {"label":" ","summary":" "}
        for node in list(self.core_subgraph.nodes()):
            self.core_subgraph.add_edge(node, self.pseudo_leaf_node)
        for node in list(self.full_graph.nodes()):
            self.full_graph.add_edge(node, self.pseudo_leaf_node)
        # leaf_nodes_training = self._intersection(train_node_ids,graph_dataset.leaf)
        # self.idx_corpus_id, self.corpus = self._construct_corpus(leaf_nodes_training)
        self.train_node_list = train_node_ids
        self.corpus, self.corpusId2nodeId = self._construct_queries(train_node_ids)
        self.trainInputLevel, self.trainlevel2nodes = self._get_level(self.core_subgraph)
        self.sample_nodes_level = self.sample_matrix_level(self.trainlevel2nodes)
        size_training = len(train_node_ids)
        train_node_ids.remove(self.root)
        sampled_nodes = random.sample(train_node_ids, int(size_training/2))
        self.train_node2pos, self.train_node2parents = self._find_insert_posistion(sampled_nodes, self.core_subgraph)
        self.trainInput = self._construct_training(sampled_nodes,self.train_node2pos,train_node_ids,sampling_method,neg_number,self.trainInputLevel)
        # down_sample = random.sample(sampled_nodes, int(len(sampled_nodes) / 4))
        self.trainInputExampleLevel = self._construct_training_levels_v1(sampled_nodes,self.train_node2pos,train_node_ids,sampling_method,neg_number)

        # self.corpusId2nodeId = self._construct_corpusId2nodeId(train_node_ids)

        #validation
        self.allLevel,self.allLevel2nodes = self._get_level(self.full_graph)
        self.valid_node_list = graph_dataset.validation_node_ids
        self.valid_queries,_ = self._construct_queries(self.valid_node_list)
        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.validation_node_ids)
        self.valid_node2pos, self.valid_node2parents = self._find_insert_posistion(graph_dataset.validation_node_ids, holdout_subgraph)
        self.val_examples = self._construct_validation(self.valid_node2pos,self.allLevel)
        # self.valInputExampleLevel = self._construct_training_levels(self.valid_node2pos, self.sample_nodes_level,
                                                                      # self.allLevel)
        self.dev_sentences1,self.dev_sentences2,self.dev_labels = self._construct_validation_levels(self.valid_node2pos)

        #test
        self.test_node_list = graph_dataset.test_node_ids
        self.test_queries,_ = self._construct_queries(self.test_node_list)
        holdout_subgraph = self._get_holdout_subgraph(graph_dataset.train_node_ids + graph_dataset.test_node_ids)
        for node in [node for node in holdout_subgraph.nodes() if
                             holdout_subgraph.out_degree(node) == 0]:
                    holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
        self.test_node2pos, self.test_node2parents  = self._find_insert_posistion(graph_dataset.test_node_ids, holdout_subgraph)
        self.test_examples = self._construct_validation(self.test_node2pos,self.allLevel)
        # self.testInputExampleLevel = self._construct_training_levels(self.test_node2pos, self.sample_nodes_level,
        #                                                               self.allLevel)
        self.all_edges = list(self._get_candidate_positions(self.core_subgraph))
        print(self.root)
        print(self.pseudo_leaf_node)
        print(8838 in self.core_subgraph.nodes)
        print(9711 in self.core_subgraph.nodes)
        print((1066, 8838) in self.all_edges)
        print((1066, 9711) in self.all_edges)
        for node, pos in self.test_node2pos.items():
            if (1066, 8838) in pos:
                print(node)
                print(pos)
        print(len(self.all_edges))
        print(len(self.core_subgraph.nodes))
        print(len(self.core_subgraph.edges))
        # exit()
        

    def get_level_list(self, l, levels):
        return [levels[elmt] for elmt in l]


    def sample_matrix_level(self,trainlevel2nodes):
        nodes_level = {}
        for level in range(-1,max(trainlevel2nodes)+1):
            nodes_in_level = trainlevel2nodes[level]
            nodes_level[level] = random.sample(nodes_in_level, int(len(nodes_in_level)/4) if len(nodes_in_level)>max(trainlevel2nodes) else len(nodes_in_level))
        return nodes_level


    def _get_level(self, graph):
        labels = {}
        node_ids = list(graph.nodes())
        level2nodes = {}
        for node in node_ids:
            label = nx.shortest_path_length(graph,self.root,node)-1
            labels[node] = label
            if label in level2nodes:
                level2nodes[label].append(node)
            else:
                level2nodes[label] = [node]
        return labels,level2nodes

    def _construct_validation(self, valid_node2pos, valLevel):
        val_examples = []
        for node in valid_node2pos:
            node_def = self.definitions[node]["summary"]
            pos_node = valid_node2pos[node]
            parents, children = zip(*pos_node)
            s_parents, s_children = set(parents), set(children)
            s_parents.update(s_children)
            for pnode in s_parents:
                pnode_def = self.definitions[pnode]["summary"]
                label_assign = round(random.uniform(0.8,1),2)
                try:
                    val_examples.append(InputExample(guid=str(node)+'_'+str(pnode),texts=[node_def, pnode_def], label=[label_assign,valLevel[node],valLevel[pnode]]))
                except:
                    pdb.set_trace()
        return val_examples


    def _construct_validation_levels(self, valid_node2pos):
        dev_sentences1 = []
        dev_sentences2 = []
        dev_labels = []
        for node in valid_node2pos:
            node_def = self.definitions[node]["summary"]
            pos_node = valid_node2pos[node]
            parents, children = zip(*pos_node)
            s_parents, s_children = set(parents), set(children)
            for pnode in s_parents:
                pnode_def = self.definitions[pnode]["summary"]
                dev_sentences1.append(node_def)
                dev_sentences2.append(pnode_def)
                dev_labels.append(1)
            for pnode in s_children:
                pnode_def = self.definitions[pnode]["summary"]
                dev_sentences1.append(node_def)
                dev_sentences2.append(pnode_def)
                dev_labels.append(0)
        return dev_sentences1,dev_sentences2,dev_labels

    def _construct_training_levels_v1(self,sampled_nodes,train_node2pos,train_node_ids,sampling_method,neg_number):
        pos_sample, neg_sample, pos_parent  = self._construct_samples(sampled_nodes,train_node2pos,train_node_ids,sampling_method,neg_number)
        train_examples = []
        core_subgraph_un = self.core_subgraph.to_undirected()
        core_subgraph_un.remove_node(self.pseudo_leaf_node)
        for node in pos_sample:
            node_def = self.definitions[node]["summary"]
            pos_node = pos_sample[node]
            parent_node = pos_parent[node]
            for posn in pos_node:
                posn_def = self.definitions[posn]["summary"]
                if posn in parent_node:
                    train_examples.append(InputExample(guid=str(node)+'_'+str(posn),texts=[node_def, posn_def], label=[1]))
                else:
                    train_examples.append(
                        InputExample(guid=str(node) + '_' + str(posn), texts=[node_def, posn_def], label=[0]))
            neg_node = neg_sample[node]
            for negn in neg_node:
                negn_def = self.definitions[negn]["summary"]
                train_examples.append(InputExample(guid=str(node)+'_'+str(negn),texts=[node_def, negn_def], label=[0]))
        return train_examples

    def _construct_training_levels_v0(self,sampled_nodes,sample_nodes_level,trainInputLevel):
        train_levels = []
        for node_s in sampled_nodes:
            level_node_s = trainInputLevel[node_s]
            node_s_def = self.definitions[node_s]["summary"]
            pos_samples = random.sample(sample_nodes_level[level_node_s-1], 50 if len(sample_nodes_level[level_node_s-1])>50 else len(sample_nodes_level[level_node_s-1]))
            neg_samples_all = []
            for k in sample_nodes_level:
                if k!=(level_node_s-1):
                    neg_samples_all.extend(sample_nodes_level[k])
            neg_samples = random.sample(neg_samples_all,len(pos_samples)*2 if len(neg_samples_all)>(len(pos_samples)*2) else len(neg_samples_all))
            pos_samples.extend(neg_samples)
            for node_l in pos_samples:
                level_node_l = trainInputLevel[node_l]
                node_l_def = self.definitions[node_l]["summary"]
                if (level_node_s - level_node_l) == 1:
                    label = 1
                # elif (level_node_s - level_node_l) == 0:
                #     label = 0
                else:
                    label = 0

                train_levels.append(InputExample(guid=str(node_s)+'_'+str(node_l),texts=[node_s_def, node_l_def], label=[label]))
        return train_levels


    def _construct_training(self,sampled_nodes,train_node2pos,train_node_ids,sampling_method,neg_number,trainInputLevel):
        pos_sample, neg_sample,_ = self._construct_samples(sampled_nodes,train_node2pos,train_node_ids,sampling_method,neg_number)
        train_examples = []
        core_subgraph_un = self.core_subgraph.to_undirected()
        core_subgraph_un.remove_node(self.pseudo_leaf_node)
        for node in pos_sample:
            node_def = self.definitions[node]["summary"]
            pos_node = pos_sample[node]
            for posn in pos_node:
                posn_def = self.definitions[posn]["summary"]
                train_examples.append(InputExample(guid=str(node)+'_'+str(posn),texts=[node_def, posn_def], label=[1.0,trainInputLevel[node],trainInputLevel[posn]]))
            neg_node = neg_sample[node]
            for negn in neg_node:
                negn_def = self.definitions[negn]["summary"]
                if sampling_method == "closest":
                    label_to_assign = 1/(nx.shortest_path_length(core_subgraph_un,node,negn))
                elif sampling_method == "closest_sign":
                    if nx.has_path(self.core_subgraph,negn,node):
                        label_to_assign = 1/(nx.shortest_path_length(core_subgraph_un,negn,node))
                    else:
                        label_to_assign = -1 / (nx.shortest_path_length(core_subgraph_un, negn, node))
                elif sampling_method == "closest_sign_square":
                    if nx.has_path(self.core_subgraph,negn,node):
                        label_to_assign = 1/(nx.shortest_path_length(core_subgraph_un,negn,node)**2)
                    else:
                        label_to_assign = -1 / (nx.shortest_path_length(core_subgraph_un, negn, node)**2)
                elif sampling_method == "closest_square":
                    label_to_assign = 1 / (nx.shortest_path_length(core_subgraph_un, node, negn)**2)
                elif sampling_method == "CSCH":
                    label_to_assign = float(csch(nx.shortest_path_length(core_subgraph_un, node, negn)))
                elif sampling_method == "sqrt":
                    label_to_assign = float(1 / (sqrt(nx.shortest_path_length(core_subgraph_un, node, negn))))
                else:
                    label_to_assign = 0.0

                train_examples.append(InputExample(guid=str(node)+'_'+str(negn),texts=[node_def, negn_def], label=[label_to_assign,trainInputLevel[node],trainInputLevel[negn]]))
        return train_examples


    def _construct_samples(self, sampled_nodes,train_node2pos,train_node_ids,sampling_method,neg_number):
        pos_sample, pos_parent = self._sample_positive(sampled_nodes,train_node2pos)
        if sampling_method.startswith("closest"):
            neg_sample = self._sample_negative_closest(sampled_nodes, pos_sample,train_node_ids,neg_number)
        else:
            neg_sample = self._sample_negative_random(sampled_nodes, pos_sample, train_node_ids,neg_number)
        return pos_sample,neg_sample, pos_parent

    def _sample_positive(self,sampled_nodes,train_node2pos):
        pos_sample = {}
        pos_parent = {}
        for node in sampled_nodes:
            try:
                parents, children = zip(*train_node2pos[node])
                s_parents, s_children = set(parents), set(children)
                pos_parent[node] = list(s_parents)
                s_parents.update(s_children)
                pos_sample[node] = list(s_parents)
            except:
                pdb.set_trace()
        return pos_sample, pos_parent

    def _sample_negative_random(self, sampled_nodes, pos_sample, train_node_ids,neg_number):
        neg_rand_sample = {}
        for node in sampled_nodes:
            s_pos = pos_sample[node]
            s_pos.append(node)
            sampling_set = list(set(train_node_ids).difference(set(s_pos)))
            neg_rand_sample[node] = random.sample(sampling_set,min(len(sampling_set),len(s_pos)*neg_number))
        return neg_rand_sample

    def _sample_negative_closest(self, sampled_nodes, pos_sample,train_node_ids,neg_number):
        neg_rand_sample = {}
        sparse_nodes = 0
        for node in sampled_nodes:
            sampling_set = set(train_node_ids)
            sampling_set.remove(node)
            path_root_leaf = list(reversed(nx.shortest_path(self.core_subgraph, source=self.root, target=node)))
            path_root_leaf.remove(self.root)
            path_root_leaf.remove(node)
            s_pos = pos_sample[node]
            if self.root in s_pos:
                s_pos.remove(self.root)
            if self.pseudo_leaf_node in s_pos:
                s_pos.remove(self.pseudo_leaf_node)
            siblings = set(path_root_leaf) - set(s_pos)
            parents = set(self.train_node2parents[node])
            for parent in parents:
                if parent!=self.root:
                    all_children = nx.descendants(self.core_subgraph,source=parent)
                    all_children.remove(node)
                    if self.pseudo_leaf_node in all_children:
                        all_children.remove(self.pseudo_leaf_node)
                    siblings.update(all_children)
            if len(siblings)<len(s_pos):
                list_sample = list(sampling_set.difference(set(s_pos)))
                neg_rand_sample[node] = random.sample(list_sample,min(len(s_pos)*neg_number,len(list_sample)))
                sparse_nodes += 1
            else:
                close_len = len(s_pos) * int(neg_number / 2)
                if len(siblings) < close_len:
                    siblings_sample = list(siblings)
                else:
                    siblings_sample = random.sample(list(siblings),close_len)
                sampling_set_filtered = sampling_set.difference(set(s_pos)).difference(set(siblings_sample))
                random_sample_neg = random.sample(sampling_set_filtered,min(len(sampling_set_filtered),close_len))
                neg_rand_sample[node] = siblings_sample + random_sample_neg
        return neg_rand_sample


    def _construct_corpus_nodes(self, queries_idx):
        return [self.definitions[elmt]['label'] +' [SEP] '+ self.definitions[elmt]['summary'] for elmt in queries_idx]

    # def _construct_corpusId2nodeId(self, queries_idx):
    #     corpusId2nodeId = {}
    #     for idx, node in enumerate(queries_idx):
    #         corpusId2nodeId[idx] = node
    #     return corpusId2nodeId

    def _construct_queries(self, queries_idx):
        def_corpus = []
        corpusId2nodeId ={}
        for idx in range(len(queries_idx)):
            def_corpus.append(self.definitions[queries_idx[idx]]['summary'])
            corpusId2nodeId[idx] = queries_idx[idx]
        assert len(queries_idx) == len(def_corpus)
        return def_corpus,corpusId2nodeId


    def _construct_corpus(self,leaf_nodes_training):
        corpus = []
        idx = 0
        idx_corpus_id = {}
        for node in leaf_nodes_training:
            path_root_leaf = list(reversed(nx.shortest_path(self.core_subgraph, source=self.root, target=node)))
            words_root_leaf = " ".join([self.definitions[elmt]['summary'] for elmt in path_root_leaf])
            idx_corpus_id[idx] = path_root_leaf
            corpus.append(words_root_leaf)
            idx += 1

        return idx_corpus_id, corpus


    def _intersection(self, lst1, lst2):
        return list(set(lst1) & set(lst2))

    def _get_holdout_subgraph(self, node_ids):
        node_to_remove = [n for n in self.full_graph.nodes if n not in node_ids]
        subgraph = self.full_graph.subgraph(node_ids).copy()
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
            for p in parents:
                for c in children:
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
                        # #LIU: 这里导致mesh数据集跑不起来，没有下面这个判断的化
                        if subgraph.in_degree(s) > 1:
                            subgraph.remove_edge(node, s)
        return subgraph

    def _get_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates

    def _find_insert_posistion(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        node2parents = {}
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
            if not parents:
                parents.add(self.root)
            position = [(p, c) for p in parents for c in children if p!=c]
            parents = [p for p in parents for c in children if p != c]
            node2pos[node] = position
            node2parents[node] = parents
        return node2pos,node2parents

