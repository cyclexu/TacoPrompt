import networkx as nx
import math
import argparse
import torch
from torch.utils.data import DataLoader
import data_process.split_data as st
import data_process.data_loader as dl
from model.sbert import SentenceTransformer, losses
from model.sbert.evaluation import EmbeddingSimilarityEvaluator
# from sentence_transformers import losses, util
# from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
# from sentence_transformers.readers import InputExample
import compute_metrics.metric as ms
from parse_config import ConfigParser
from model.utils import PPRPowerIteration
import pickle as pkl



torch.manual_seed(0)
args = argparse.ArgumentParser(description='Training taxonomy expansion model')
args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
config = ConfigParser(args)
args = args.parse_args()

saving_path = config['saving_path']
name = config['name']
data_path = config['data_path']
sampling_method = config['sampling']
neg_number = config['neg_number']
partition_pattern = config['partition_pattern']
seed = config['seed']
batch_size = config['batch_size']
epochs = config['epochs']
alpha = config['alpha']

taxonomy = dl.TaxoDataset(name,data_path,raw=True,partition_pattern=partition_pattern,seed=seed)
data_prep = st.Dataset(taxonomy,sampling_method,neg_number,seed)
model_name = config['model_name']

device = "cuda" if torch.cuda.is_available() else "cpu"
target_device = torch.device(device)

if torch.cuda.is_available():
    model = SentenceTransformer.SentenceTransformer(model_name, device='cuda')
else:
    model = SentenceTransformer.SentenceTransformer(model_name)

g = torch.Generator()
g.manual_seed(0)


nodeIdsCorpus =[data_prep.corpusId2nodeId[idx] for idx in data_prep.corpusId2nodeId]
core_graph = data_prep.core_subgraph.copy()
core_graph.remove_node(data_prep.pseudo_leaf_node)
nodes_core_subgraph = list(core_graph.nodes)
assert nodes_core_subgraph == nodeIdsCorpus
propagation = PPRPowerIteration(nx.adjacency_matrix(core_graph), alpha=alpha, niter=10).to(target_device)


# Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(data_prep.trainInput, shuffle=True, batch_size=batch_size)
warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1) #10% of train data for warm-up
train_loss = losses.CosineSimilarityLoss(model)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(data_prep.val_examples, name='sts-dev')
# Tune the model
# save_path = "data/SemEval-Food/0927_225436"




# model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, evaluation_steps=warmup_steps, epochs=epochs,
#           warmup_steps=warmup_steps, output_path=save_path, save_best_model=True)
# model_path = 'data/SemEval-Food/test_2023_8_26_20_23'
# model_path = 'data/SemEval-Food/test_2023_8_26_18_54'
# model.save(model_path)
# best_score = -100.0  
# epoch_not_improve = 0
# best_epoch = 0
# EARLY_STOP = 10
# for epoch in range(epochs):
#     if epoch_not_improve >= EARLY_STOP:
#         break
#     model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=warmup_steps)
#     score = evaluator(model, output_path=None)
#     if score > best_score:
#         print(f"Epoch {epoch+1}: improve (score: {score})")
#         model.save(str(config.save_dir))
#         best_score = score
#         epoch_not_improve = 0
#         best_epoch = epoch
#     else:
#         print(f"Epoch {epoch+1}: not improve (score: {score})")
#         epoch_not_improve += 1


model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, evaluation_steps=100, epochs=epochs,
          warmup_steps=warmup_steps, output_path=str(config.save_dir), save_best_model=True)

model = SentenceTransformer.SentenceTransformer(str(config.save_dir))
# model = SentenceTransformer.SentenceTransformer('data/SemEval-Food/30-Sep-2023 (22:08:00.090980)')
# model = SentenceTransformer.SentenceTransformer(data/mesh/29-Sep-2023 (17:34:44.067334)')
# model = SentenceTransformer.SentenceTransformer('data/SemEval-Verb/29-Sep-2023 (16:32:14.471060)')
corpus_embeddings = model.encode(data_prep.corpus, convert_to_tensor=True, show_progress_bar=True)
preds = propagation(corpus_embeddings,torch.tensor(range(len(nodeIdsCorpus)),device=target_device))

print(len(data_prep.corpusId2nodeId))

# all_targets_val, all_predictions_val, all_scores_val, edges_predictions_val, all_edges_scores_val = ms.compute_prediction(data_prep.core_subgraph.edges,data_prep.pseudo_leaf_node, data_prep.valid_queries,corpus_embeddings,model,data_prep.valid_node_list,data_prep.valid_node2pos,data_prep.corpusId2nodeId)
# ms.save_results_with_ranks(str(config.save_dir)+'/',all_targets_val, edges_predictions_val,"eval_val",leaf_node=data_prep.pseudo_leaf_node)
# all_targets_test, all_predictions, all_scores_test, edges_predictions_test, all_edges_scores_test  = ms.compute_prediction(data_prep.core_subgraph.edges, data_prep.pseudo_leaf_node, data_prep.test_queries,corpus_embeddings,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
# ms.save_results_with_ranks(str(config.save_dir)+'/',all_targets_test, edges_predictions_test,"eval_test",leaf_node=data_prep.pseudo_leaf_node)


# all_targets_val_ppr, all_predictions_val_ppr, all_scores_val_ppr, edges_predictions_val_ppr, all_edges_scores_val_ppr = ms.compute_prediction(data_prep.core_subgraph.edges,data_prep.pseudo_leaf_node, data_prep.valid_queries,preds,model,data_prep.valid_node_list,data_prep.valid_node2pos,data_prep.corpusId2nodeId)
# ms.save_results_with_ranks(str(config.save_dir)+'/',all_targets_val_ppr, edges_predictions_val_ppr,"eval_val_ppr",leaf_node=data_prep.pseudo_leaf_node)
# with open()
# print(data_prep.test_queries)

print(len(data_prep.all_edges))
print(len(data_prep.corpusId2nodeId))
# all_targets_test_ppr, all_predictions_ppr, all_scores_test_ppr, edges_predictions_test_ppr, all_edges_scores_test_ppr  = ms.compute_prediction(data_prep.all_edges, data_prep.pseudo_leaf_node, data_prep.test_queries,preds,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
# all_targets_test_ppr, all_predictions_ppr, all_scores_test_ppr, edges_predictions_test_ppr, all_edges_scores_test_ppr  = ms.compute_prediction(data_prep.all_edges, data_prep.pseudo_leaf_node, data_prep.test_queries,preds,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
# ms.save_results_with_ranks(str(config.save_dir)+'/',all_targets_test_ppr, edges_predictions_test_ppr,"eval_test_ppr",leaf_node=data_prep.pseudo_leaf_node)
all_ranks, leaf_ranks, nonleaf_ranks  = ms.compute_prediction_efficient(data_prep.all_edges, data_prep.pseudo_leaf_node, data_prep.test_queries,preds,model,data_prep.test_node_list,data_prep.test_node2pos,data_prep.corpusId2nodeId)
ms.save_results_with_ranks_efficient(all_ranks, leaf_ranks, nonleaf_ranks)

# print(best_epoch)