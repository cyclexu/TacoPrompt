import pdb
import time
import torch
import numpy as np
import itertools
import re
from sentence_transformers import util
import pickle as pkl


from pathlib import Path

def save_results(saving_path,all_targets, all_predictions,name_file):
    Path(saving_path).mkdir(parents=True, exist_ok=True)
    f = open(saving_path + name_file + ".csv", "a")
    line_metric = "\nprec@1,prec@5,prec@10,recall@1,recall@5,recall@10,MR,MRR\n"
    f.write(line_metric)
    relevance = get_relevance(all_targets, all_predictions)
    prec1, prec5, prec10 = compute_precision(relevance, 1), compute_precision(relevance, 5), compute_precision(
        relevance, 10)
    rec1, rec5, rec10 = compute_recall(relevance, 1), compute_recall(relevance, 5), compute_recall(relevance,
                                                                                                            10)
    mr = micro_mr(relevance)
    mrr = compute_mrr(relevance)
    line = "{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(prec1, prec5, prec10, rec1, rec5, rec10,
                                                                            mr,mrr)
    f.write(line)
    f.close()
    print(line_metric,line)
    return line_metric,line

def save_results_with_ranks(saving_path,all_targets, all_predictions,name_file,leaf_node):
    # Path(saving_path).mkdir(parents=True, exist_ok=True)
    # f = open(saving_path + name_file + ".csv", "a")
    # line_metric = "\nprec@1,prec@5,prec@10,recall@1,recall@5,recall@10,MR,MRR\n"
    # f.write(line_metric)
    print(name_file)
    all_ranks, leaf_ranks, non_leaf_ranks = get_all_ranks(all_targets, all_predictions,leaf_node)
    print(all_ranks)
    print(leaf_ranks)
    print(non_leaf_ranks)
    metrics = [macro_mr, micro_mr_v1, mrr_scaled_10, hit_at_1, hit_at_5, hit_at_10, hit_at_50, hit_at_100, precision_at_1, precision_at_5, precision_at_10, real_hit_at_1, real_hit_at_5, real_hit_at_10, real_hit_at_50, real_hit_at_100]
    total_metrics = {metric.__name__ : metric(all_ranks) for metric in metrics}
    leaf_metrics = {f'leaf_{metric.__name__}' : metric(leaf_ranks) for metric in metrics}
    non_leaf_metrics = {f'nonleaf_{metric.__name__}' : metric(non_leaf_ranks) for metric in metrics}
    print(total_metrics)
    for k, v in total_metrics.items():
        print(f'{k}:{v}')
    for k, v in leaf_metrics.items():
        print(f'{k}:{v}')
    for k, v in non_leaf_metrics.items():
        print(f'{k}:{v}')

def save_results_with_ranks_efficient(all_ranks, leaf_ranks, nonleaf_ranks):
    # Path(saving_path).mkdir(parents=True, exist_ok=True)
    # f = open(saving_path + name_file + ".csv", "a")
    # line_metric = "\nprec@1,prec@5,prec@10,recall@1,recall@5,recall@10,MR,MRR\n"
    # f.write(line_metric)
    # print(name_file)
    # all_ranks, leaf_ranks, non_leaf_ranks = get_all_ranks(all_targets, all_predictions,leaf_node)
    print(all_ranks)
    # print(leaf_ranks)
    # print(nonleaf_ranks)
    metrics = [macro_mr, micro_mr_v1, mrr_scaled_10, hit_at_1, hit_at_5, hit_at_10, hit_at_50, hit_at_100, precision_at_1, precision_at_5, precision_at_10, real_hit_at_1, real_hit_at_5, real_hit_at_10, real_hit_at_50, real_hit_at_100]
    total_metrics = {metric.__name__ : metric(all_ranks) for metric in metrics}
    leaf_metrics = {f'leaf_{metric.__name__}' : metric(leaf_ranks) for metric in metrics}
    non_leaf_metrics = {f'nonleaf_{metric.__name__}' : metric(nonleaf_ranks) for metric in metrics}
    print(total_metrics)
    for k, v in total_metrics.items():
        print(f'{k}:{v}')
    for k, v in leaf_metrics.items():
        print(f'{k}:{v}')
    for k, v in non_leaf_metrics.items():
        print(f'{k}:{v}')
        
        
def compute_prediction_efficient(edges,leaf_node, queries,corpus_embeddings,model,node_list,node2positions,corpusId2nodeId):
    '''
    edges: core_subgraph中的所有边
    leaf_node: 伪叶节点
    queries: 验证集或者测试集中划分出来的叶子
    corpus_embedings: 自监督学习出来的节点embeddings.
    model: 训练好的模型。
    node_list: 所有验证集或者测试集中的节点
    node2positions: 验证集或者测试集中，节点对应的位置
    corpusId2nodeId: def的编号和节点编号的对应。
    '''
    top_k = len(corpus_embeddings)
    all_targets = []
    all_predictions = []
    all_scores = []
    all_edges_scores,edges_prediction = [],[]
    edges_2darray = np.array([*list(edges)])
    parent = edges_2darray[:,0]
    children = edges_2darray[:,1]
    query_dict = {}
    # print(node2positions)
    
    """
    """
    
    query_emb = model.encode(queries,convert_to_tensor=True,
                            show_progress_bar=False)
    score = util.cos_sim(query_emb,corpus_embeddings)
    # print(parent.shape,children.shape,score.shape)
    # exit()
    
    
    children_leaf = np.where(children == leaf_node)[0]
    children[children_leaf] = parent[children_leaf]
    nodeid2copus_id = {v:k for k,v in corpusId2nodeId.items()}
    parent_idx = [nodeid2copus_id[i] for i in parent]
    children_idx = [nodeid2copus_id[i] for i in children]
    # print(len(children_leaf))
    # exit()
    parent_idx = torch.tensor(parent_idx).long().to('cuda')
    children_idx = torch.tensor(children_idx).long().to('cuda')
    # children_leaf = torch.where(children_idx == leaf_node)[0]
    # children_idx[children_leaf] = parent_idx[children_leaf]
    # # print(children_idx)
    # # print(len(torch.where(children_idx == leaf_node)[0]))
    # # # print(leaf_node)
    # # exit()
    # nodeid2copus_id = {v:k for k,v in corpusId2nodeId.items()}
    # parent_idx = [nodeid2copus_id[i] for i in parent]
    # children_idx = [nodeid2copus_id[i] for i in children]
    
    parent_score = score[:,parent_idx]
    children_score = score[:,children_idx]
    scores = (parent_score + children_score)/2
    # print(scores.shape)
    # print(children_score)
    # exit()
    ranks = torch.argsort(scores,dim=1,  descending=True)
    top100ranks = ranks[:,:100]
    top100_dict = {}
    for idx, query in enumerate(queries):
        query_id = node_list[idx]
        query_rank = top100ranks[idx,:]
        query_rank = query_rank.tolist()
        query_rank = [edges[pos_id] for pos_id in query_rank]
        for i,pos in enumerate(query_rank):
            if query_rank[i][1] == leaf_node:
                query_rank[i] = (query_rank[i][0],leaf_node + 1)
        top100_dict[query_id] = query_rank
    with open('complete_top100_dict_food_20epoch.pkl','wb') as f:
        pkl.dump(top100_dict,f)
    
    # print(ranks)
    # print(edges)
    
    target_positions = [node2positions[node_list[idx]] for idx, query in enumerate(queries)]
    # print(target_positions)
    target_positions_idx = [[edges.index(pos) for pos in query2pos if pos in edges] for query2pos in target_positions]
    # print(target_positions_idx)
    # print(edges[5483])
    
    all_ranks = []
    leaf_queries = []
    all_ranks = []
    leaf_ranks, nonleaf_ranks = [], []
    for idx, query in enumerate(queries):
        query_id = node_list[idx]
        poses = node2positions[query_id]
        flag = True
        for pos in poses:
            if pos[1] != leaf_node:
                flag = False
                break
        if flag:
            leaf_queries.append(query_id)
        
    for idx, query in enumerate(queries):
        query_rank = ranks[idx].cpu().numpy().tolist()
        query_id = node_list[idx]
        true_rank = [query_rank.index(pos)+1 for pos in target_positions_idx[idx]]
        all_ranks.append(true_rank)
        if query_id in leaf_queries:
            leaf_ranks.append(true_rank)
        else:
            nonleaf_ranks.append(true_rank)
        
    return all_ranks, leaf_ranks, nonleaf_ranks

def compute_prediction(edges,leaf_node, queries,corpus_embeddings,model,node_list,node2positions,corpusId2nodeId):
    '''
    edges: core_subgraph中的所有边
    leaf_node: 伪叶节点
    queries: 验证集或者测试集中划分出来的叶子
    corpus_embedings: 自监督学习出来的节点embeddings.
    model: 训练好的模型。
    node_list: 所有验证集或者测试集中的节点
    node2positions: 验证集或者测试集中，节点对应的位置
    corpusId2nodeId: def的编号和节点编号的对应。
    '''
    top_k = len(corpus_embeddings)
    all_targets = []
    all_predictions = []
    all_scores = []
    all_edges_scores,edges_prediction = [],[]
    edges_2darray = np.array([*list(edges)])
    parent = edges_2darray[:,0]
    children = edges_2darray[:,1]
    query_dict = {}
    # print(node2positions)
    
    for idx, query in enumerate(queries):
        try:
            query_id = node_list[idx] #取得query_id
            target_positions = node2positions[query_id] #这个query对应的位置
            all_targets.append(target_positions) #
            question_embedding = model.encode(query, convert_to_tensor=True) #query的desc的embedding
            hits_score = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k) 
            # hits_score 是query对于corpus所有描述做相似度对比，似乎还有排序
            hits = [corpusId2nodeId[hit['corpus_id']] for hit in
                    hits_score[0]]  # Get the hits for the first query
            scores = [hit['score'] for hit in hits_score[0]]
    # ###############################################################################################################     
    # #用于计算节点之间的相似度
    #         tmp_dic = {k:v for k,v in zip(hits,scores)}
    #         query_dict[query_id] = tmp_dic
            
    #     except:
    #         assert "wrong"
    # return query_dict
    # ################################################################################################################
    
            hits.append(leaf_node)
            scores.append(2)
            scores_arr = np.array(scores)
            ind_parents = np.where(hits==parent[:,None])[1]
            ind_child = np.where(hits==children[:,None])[1]
            scores_2darray = np.append([scores_arr[ind_parents]],[scores_arr[ind_child]],axis=0).T
            args_leaf = np.where(scores_2darray[:,1]==2)
            scores_2darray[args_leaf,1] = scores_2darray[args_leaf,0]
            # for id_x,x in enumerate(query_pred):
            #     scores_2darray[np.where(edges_2darray==x)]=scores_pred[id_x]
            scores_mean = scores_2darray.mean(axis=1)
            sorting_args = np.argsort(scores_mean)[::-1]
            edges_prediction.append(edges_2darray[sorting_args,:])
            all_edges_scores.append(scores_mean[sorting_args])
            all_predictions.append(hits)
            all_scores.append(scores)
        except:
            pdb.set_trace()
    print(np.array(all_edges_scores).shape)
    return all_targets, all_predictions, all_scores, edges_prediction, all_edges_scores

def get_all_ranks(all_target, pred_pos, leaf_node):
    all_ranks = []
    leaf_ranks = []
    non_leaf_ranks = []
    pred_pos_np = np.array(pred_pos)
    for idx, target_parents in enumerate(all_target):
        flag = True
        for pos in target_parents:
            if pos[1] != leaf_node:
                flag = False
                break
        ranks = []
        for (parent,child) in target_parents:
            identify_idx = np.where((pred_pos_np[idx] == (parent,child)).all(axis=1))[0]
            if len(identify_idx)>0:
                posIdx = identify_idx[0]
            else:
                posIdx = np.where(pred_pos_np[idx] == (parent,child))[0][0]
            rank = posIdx + 1
            ranks.append(rank)
        # print(len(ranks),len(target_parents))
        all_ranks.append(ranks)
        if flag:
            leaf_ranks.append(ranks)
        else:
            non_leaf_ranks.append(ranks)
    return all_ranks,leaf_ranks,non_leaf_ranks

def get_relevance(all_target, pred_pos):
    relevance = []
    pred_pos_np = np.array(pred_pos)
    for idx, target_parents in enumerate(all_target):
        relevance.append([0]*len(pred_pos[idx]))
        for (parent,child) in target_parents:
            identify_idx = np.where((pred_pos_np[idx] == (parent,child)).all(axis=1))[0]
            if len(identify_idx)>0:
                posIdx = identify_idx[0]
            else:
                posIdx = np.where(pred_pos_np[idx] == (parent,child))[0][0]  
            relevance[idx][posIdx] = 1
    return np.array(relevance)

def micro_mr(relevance):
    ranks = [np.nonzero(t)[0] for t in relevance]
    ranks_l =[elm[0] for elm in ranks]
    micro_mr = sum(ranks_l)/len(ranks_l)
    return micro_mr

def compute_recall(relevance, r=10):
    true_position_in_top_r = np.any(relevance[:, :r], axis=1)
    return np.mean(true_position_in_top_r)

def compute_precision(relevance, r=10):
    true_position_in_top_r = np.any(relevance[:, :r], axis=1)
    return 1.0 * np.sum(true_position_in_top_r) / (len(true_position_in_top_r)*r)


def compute_mrr(relevance, r=10):
    """Compute the mean reciprocal rank of a set of queries.

    relevance is a numpy matrix; each row contains the "relevance"
    of the predictions (= 0 or 1) made for each query.

    predictions are ranked in decreasing order of relevence.
    relevance[:, :15] are the top 15 most relevent predictions.

    The first non-zero entry of each row is the lowest ranked correct
    prediction. The reciprocal of this rank is the reciprocal rank.
    The mean of the reciprocal rank over all queries is returned as
    a percentage.

    Example:

        relevance = [[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 0]]

        ranks = [[2], [1], []]  # this is 0-based

        mrrs = [1/(2+1), 1/(1+1), 0] = [1/3, 1/2, 0]
    """
    ranks = [np.nonzero(t)[0] for t in relevance[:, :]]
    mrrs = [1.0/(rank[0] + 1) if len(rank) > 0 else 0.0
            for rank in ranks]
    return 100.0 * np.mean(mrrs)




def macro_mr(all_ranks):
    macro_mr = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return macro_mr


def micro_mr_v1(all_ranks):
    micro_mr = np.array(list(itertools.chain(*all_ranks))).mean()
    return micro_mr


def hit_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(rank_positions)


def hit_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / len(rank_positions)


def hit_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / len(rank_positions)

def hit_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / len(rank_positions)

def hit_at_50(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 50)
    return 1.0 * hits / len(rank_positions)

def hit_at_100(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 100)
    return 1.0 * hits / len(rank_positions)


def precision_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(all_ranks)


def precision_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / (len(all_ranks) * 3)


def precision_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / (len(all_ranks) * 5)


def precision_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / (len(all_ranks) * 10)

def real_hit_at_1(all_ranks):
    hits = 0
    for ranks in all_ranks:
        for rank in ranks:
            if rank <= 1:
                hits += 1
                break
    return 1.0 * hits / len(all_ranks)


def real_hit_at_5(all_ranks):
    hits = 0
    for ranks in all_ranks:
        for rank in ranks:
            if rank <= 5:
                hits += 1
                break
    return 1.0 * hits / len(all_ranks)


def real_hit_at_10(all_ranks):
    hits = 0
    for ranks in all_ranks:
        for rank in ranks:
            if rank <= 10:
                hits += 1
                break
    return 1.0 * hits / len(all_ranks)

def real_hit_at_50(all_ranks):
    hits = 0
    for ranks in all_ranks:
        for rank in ranks:
            if rank <= 50:
                hits += 1
                break
    return 1.0 * hits / len(all_ranks)

def real_hit_at_100(all_ranks):
    hits = 0
    for ranks in all_ranks:
        for rank in ranks:
            if rank <= 100:
                hits += 1
                break
    return 1.0 * hits / len(all_ranks)


def mrr_scaled_10(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / 10)
    return (1.0 / scaled_rank_positions).mean()