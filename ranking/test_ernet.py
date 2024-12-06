import faiss, os
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import visualizations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
 
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

def test(args, eval_ds, model, er_model):
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    # del all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    # 1-st end
    if args.dataset_name != "msls":
        soft_positives_per_database = eval_ds.get_positives_database()
    else:
        soft_positives_per_database = None
        
    similarity_matrix = cosine_similarity(database_features)
    absolute_positives_per_database = get_absolute_positives(similarity_matrix, soft_positives_per_database)
    vis_sim(similarity_matrix)
    
    # Visualization of database_utm

    new_predictions = []
    weights = [0.5, 0.125, 0.125, 0.125, 0.125]

    for query_idx, prediction in enumerate(predictions):
        
        query_feature = queries_features[query_idx]
        embodied_candidates = [absolute_positives_per_database[pre] for pre in prediction[:RANK]]

        embodied_features = sim_weighted_feature(query_feature, [database_features[cand] for cand in embodied_candidates], query_idx, CORRECT_NUMS)
        cosine_similarities = (np.dot(embodied_features, query_feature) / (
                        np.linalg.norm(embodied_features, axis=1) * np.linalg.norm(query_feature)))
        distances = 1-cosine_similarities

        ranked_indices = np.argsort(distances)
        ranked_prediction = prediction[:RANK][ranked_indices]

        unranked_predictions = prediction[RANK:]
        new_prediction = np.concatenate((ranked_prediction, unranked_predictions))

        new_predictions.append(new_prediction)

    predictions = np.array(new_predictions)

    if args.dataset_name == "msls_challenge":
        fp = open("msls_challenge.txt", "w")
        for query in range(eval_ds.queries_num):
            query_path = eval_ds.queries_paths[query]
            fp.write(query_path.split("@")[-1][:-4]+' ')
            for i in range(20):
                pred_path = eval_ds.database_paths[predictions[query,i]]
                fp.write(pred_path.split("@")[-1][:-4]+' ')
            fp.write("\n")
        fp.write("\n")

    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.2f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)


    return recalls, recalls_str