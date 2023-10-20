from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
from data_loader.data_loaders import UnifiedDataLoader
from data_loader.dataset import *
import argparse
import pickle
from tqdm import tqdm


# top k similar train nodes
def get_k_sim(q, k=5):
    concepts = query2sim[q][:k]
    return concepts


def get_k_sim_double(q, k=1):
    concepts = []
    for sim_c in query2sim[q]:
        if train2type[dataset.id2taxon[sim_c]]:
            concepts.append(sim_c)
            if len(concepts) == k:
                break

    for sim_c in query2sim[q]:
        if not train2type[node_id2taxon[sim_c]]:
            concepts.append(sim_c)
            if len(concepts) == 2 * k:
                break

    return concepts


# pos+neg，[p, c, yes/no, yes/no, yes/no]
def get_samples(concept, negative_size):
    samples = []
    query = dataset.id2taxon[concept]

    pos_pos = random.choice(dataset.node2pos[query])
    if negative_size:
        neg_pos = dataset._get_exactly_k_negatives(query, negative_size)

    p, c = pos_pos
    sample = (p.description, c.description, "yes", "yes", 'yes')
    samples.append(sample)
    if negative_size:
        for p, c in neg_pos:
            if c is dataset.pseudo_leaf_node:
                par_ans = "yes" if p in dataset.node2parents[query] else "no"
                chl_ans = 'no'
            else:
                par_ans = "yes" if p in dataset.node2parents[query] else "no"
                chl_ans = "yes" if c in dataset.node2children[query] else "no"
            sample = (p.description, c.description, par_ans, chl_ans, 'no')
            samples.append(sample)
    return samples


# generate examples
def get_examples(q, query2samples):
    example_prompt = "Given Parent: {}, Child: {} and Query: {} " \
                     "Taxonomy Completion: Parent is {}, child is {} and the final prediction is {}\n"
    examples = ""
    for q_samples in query2samples[q]:
        for sample in q_samples:
            examples += example_prompt.format(sample[0], sample[1], q.description,
                                              sample[2], sample[3], sample[4])
    return examples


# evaluate <q, p, c>
def get_score(q, p, c, examples):
    prompt_step_1 = "{} \n\n Given Parent: {}, Child: {} and Query: {} Taxonomy Completion: Parent is"
    # subtask p
    query_examples = examples[q]
    c_desc = c.description
    step_1 = prompt_step_1.format(query_examples, p.description, c_desc, q.description)
    inputs = tokenizer(step_1, return_tensors="pt").to(device)
    probability_values = model(inputs.input_ids).logits[:, -1, :]
    if probability_values[0][yes_token_id] >= probability_values[0][no_token_id]:
        ans_1 = 'yes'
    else:
        ans_1 = 'no'

    # subtask c
    step_2 = step_1 + " {}, child is".format(ans_1)
    inputs = tokenizer(step_2, return_tensors="pt").to(device)
    probability_values = model(inputs.input_ids).logits[:, -1, :]
    if probability_values[0][yes_token_id] >= probability_values[0][no_token_id]:
        ans_2 = 'yes'
    else:
        ans_2 = 'no'

    # final task
    step_3 = step_2 + " {} and the final prediction is".format(ans_2)
    inputs = tokenizer(step_3, return_tensors="pt").to(device)
    probability_values = model(inputs.input_ids).logits[:, -1, :]
    score = probability_values[0][yes_token_id] - probability_values[0][no_token_id]

    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-td', '--taxo-file-directory', default="SemEval-Food", type=str,
                    help='data directory (default: SemEval-Food)')
    ap.add_argument('-tn', '--taxo-name', default="semeval_food", type=str,
                    help='taxonomy name (default: semeval_food)')
    ap.add_argument('-tk', '--top-k', default=50, type=int, help='retrieval number (default: 50)')
    ap.add_argument('-ps', '--positive-sampling', default=1, type=int, help='positive examples number (default: 1)')
    ap.add_argument('-ns', '--negative-sampling', default=1, type=int, help='negative examples number (default: 1)')
    ap.add_argument('-ds', '--do-sample', default=True, type=bool, help='if re-sample examples (default: True)')

    args = vars(ap.parse_args())

    # processed data
    taxo_dir = args['taxo_file_directory']
    taxo_name = args['taxo_name']
    top_k = args['top_k']
    pos_sample = args['positive_sampling']
    neg_sample = args['negative_sampling']
    do_sample = args['do_sample']

    dataloader = UnifiedDataLoader('/data/{}/{}.pickle.bin'.format(taxo_dir, taxo_name), taxo_name)
    dataset = dataloader.dataset
    pickle_path = '/data/{}/{}.pickle.bin'.format(taxo_dir, taxo_name)
    graph_pickle_path = '/data/{}/subgraphs.pickle'.format(taxo_dir)
    print('loading pickled dataset')
    with open(pickle_path, "rb") as fin:
        data = pickle.load(fin)
        name = data["name"]
        taxonomy = data['taxonomy']
        node_id2taxon = data['id2taxon']
        taxon2node_id = data['taxon2id']
        vocab = data["vocab"]
        train_node_ids = data["train_node_ids"]
        validation_node_ids = data["validation_node_ids"]
        test_node_ids = data["test_node_ids"]
    with open(graph_pickle_path, 'rb') as f:
        graphs = pickle.load(f)
        core_subgraph = graphs['core_subgraph']
        pseudo_leaf_node = graphs['pseudo_leaf_node']
        valid_holdout_subgraph = graphs['valid_subgraph']
        valid_node2pos = graphs['valid_node2pos']
        test_holdout_subgraph = graphs['test_subgraph']
        test_node2pos = graphs['test_node2pos']

    # construct vocab
    existing_name = {}
    vocab = []
    norm_vocab = []
    for node_id, taxon in node_id2taxon.items():
        if taxo_name == 'semeval_food':
            name = taxon.norm_name.lower()
        elif taxo_name == 'wordnet_verb':
            norm_name = taxon.norm_name
            name = norm_name.split('||')[0]
            name = ' '.join(name.split('_'))
            # name = name + '#' + norm_name[-2:]
        elif taxo_name == 'mesh':
            name = taxon.description[:taxon.description.find(" is")].lower()
        vocab.append(name)
    vocab.append('root#01')  # root
    node_names = set(vocab)

    # cosine similarity between query and train
    # query_id: [train_id1, train_id2, ...], descending order
    with open('prepared_data/{}/sim_test2train.pkl'.format(taxo_dir), 'rb') as f:
        query2sim = pickle.load(f)

    # type is used for selective examples construction
    # query type: if leaf
    query2type = {}
    for query, poses in test_node2pos.items():
        flag = True
        for pos in poses:
            if pos[1] != dataset.pseudo_leaf_node:
                flag = False
                break
        query2type[query] = flag

    # train type
    train2type = {}
    for query, poses in dataset.node2pos.items():
        flag = True
        for pos in poses:
            if pos[1] != dataset.pseudo_leaf_node:
                flag = False
                break
        train2type[query] = flag

    # original ground-truth number
    gt_len = 0
    for poses in test_node2pos.values():
        gt_len += len(poses)

    # top-100 retrieval list
    # query_id: [(candidate_p_id, candidate_c_id), ...]
    with open('prepared_data/{}/retrieval_top100_dict.pkl'.format(taxo_dir), 'rb') as f:
        query2top100pos = pickle.load(f)

    # trunc to k
    print("trunc query2top100pos to query2retrieval...")
    query2retrieval = dict()
    for q in query2top100pos.keys():
        query2retrieval[q] = query2top100pos[q][:top_k]

    # get samples
    if do_sample:
        # sample from scratch
        print('generating samples')
        query2samples = {}
        for q in test_node2pos.keys():
            sim_concepts = get_k_sim(dataset.taxon2id[q], k=pos_sample)
            query2samples[q] = []
            for concept_id in sim_concepts:
                samples = get_samples(concept_id, negative_size=neg_sample)
                query2samples[q].append(samples)
        pickle_path = "samples/{}_samples.pickle".format(taxo_name)
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'query2samples': query2samples,
                'sim_concepts': sim_concepts
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # fixed samples (exp using different prompt, llama, ...)
        pickle_path = "samples/{}_samples.pickle".format(taxo_name)
        print('loading pickled samples')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
            query2samples = data['query2samples']
            sim_concepts = data['sim_concepts']

    # get examples
    examples = {}
    for query in test_node2pos.keys():
        examples[query] = get_examples(query, query2samples, example_number=pos_sample, negative_size=neg_sample)

    # ground-truth
    query2ground_truth = {}
    for query in test_node2pos.keys():
        query_id = taxon2node_id[query]
        query_ground_truth = set([(dataset.taxon2id[gt[0]], dataset.taxon2id[gt[1]]) for gt in test_node2pos[query]])
        query2ground_truth[query_id] = query_ground_truth

    # retrieval gt idx
    query_gt_idx_in_top_k = {}
    for query in test_node2pos.keys():
        query_gt_idx_in_top_k[query] = []
        query_id = taxon2node_id[query]
        query_ground_truth = query2ground_truth[query_id]
        query_top_k = query2retrieval[query_id]
        for gt in query_ground_truth:
            if gt in query_top_k:
                query_gt_idx_in_top_k[query].append(query_top_k.index(gt))

    # Llama model
    device = torch.device("cuda")
    model = LlamaForCausalLM.from_pretrained("/root/autodl-tmp/Llama-1-7b-hf").to(device)
    tokenizer = LlamaTokenizer.from_pretrained("/root/autodl-tmp/Llama-1-7b-hf")

    # yes/no token.
    yes_token_id = tokenizer.convert_tokens_to_ids(['▁yes'])[0]
    no_token_id = tokenizer.convert_tokens_to_ids(['▁no'])[0]

    # evaluate
    rank_lists = []
    leaf_rank_lists = []
    nonleaf_rank_lists = []

    with torch.no_grad():
        for query in tqdm(test_node2pos.keys(), desc='querying Llama...'):
            score = []
            query_id = taxon2node_id[query]
            if len(query_gt_idx_in_top_k[query]) == 0:
                # dummy for not exist
                rank_list = [-1]
            else:
                query_top_k = query2retrieval[query_id]
                for p_id, c_id in query_top_k:
                    # score.append(get_score(query, node_id2taxon[p_id], node_id2taxon[c_id]))
                    score.append(get_score(query, dataset.id2taxon[p_id], dataset.id2taxon[c_id], examples))
                scoce_argsort = torch.argsort(torch.tensor(score))
                rank_list = [scoce_argsort[gt_idx].item() + 1 for gt_idx in query_gt_idx_in_top_k[query]]
            rank_lists.append(rank_list)
            if query2type[query]:
                leaf_rank_lists.append(rank_list)
            else:
                nonleaf_rank_lists.append(rank_list)
            print(rank_lists, leaf_rank_lists, nonleaf_rank_lists)

    # save rank list
    print("saving rank_lists...")
    pickle_path = "saved_rank_lists/{}_rank_lists.pickle".format(taxo_name)
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'rank_lists': rank_lists,
            'leaf_rank_lists': leaf_rank_lists,
            'nonleaf_rank_lists': nonleaf_rank_lists,
            'gt_len': gt_len
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    import eval
    eval.evaluate(pickle_path)