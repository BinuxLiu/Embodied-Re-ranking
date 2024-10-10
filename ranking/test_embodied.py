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
 

# Pitts: 1, 4, 10
# MSLS:  5, 5, 10
CORRECT_NUMS = 4

WEIGHT = 0.6

# def weighted_feature(query_feature, database_features, query_idx):

#     embodied_features = [np.zeros_like(db[0]) for db in database_features]

#     weights_sum = None

#     for i in range(1, CORRECT_NUMS + 1):

#         cosine_similarities = []
#         for db in database_features:
#             if i < len(db):
#                 cosine_similarity = np.dot(db[i], query_feature) / (np.linalg.norm(db[i]) * np.linalg.norm(query_feature))
#             else:
#                 cosine_similarity = 0
#             cosine_similarities.append(cosine_similarity)

#         cos_distances = 1 - np.array(cosine_similarities)
#         weights = cos_distances / sum(cos_distances)

#         for j, db in enumerate(database_features):
#             if i < len(db):
#                 embodied_features[j] += weights[j] * db[i]

#         if weights_sum is None:
#             weights_sum = weights
#         else:
#             weights_sum += weights


#     embodied_features = [embodied_features[i] + (1-weights_sum[i]) * db[0] for i, db in enumerate(database_features)]

#     return embodied_features

def weighted_feature(query_feature, database_features, query_idx):

    embodied_features = [np.zeros_like(db[0]) for db in database_features]

    for db_idx, db in enumerate(database_features):

        cosine_similarities = []

        for i in range(CORRECT_NUMS+1):
            if i < len(db):
                cosine_similarity = np.dot(db[i], query_feature) / (np.linalg.norm(db[i]) * np.linalg.norm(query_feature))
            else:
                cosine_similarity = 0
            cosine_similarities.append(cosine_similarity)
        print(cosine_similarities)
        # cos_distances = 1 - np.array(cosine_similarities)
        weights = cosine_similarities / sum(cosine_similarities)

        for i in range(CORRECT_NUMS+1):
            if i < len(db):
                embodied_features[db_idx] += weights[i] * db[i]
            else:
                embodied_features[db_idx] += 0 

        embodied_features[db_idx] = (1-WEIGHT) * embodied_features[db_idx] + WEIGHT * db[0]

    return embodied_features


def get_absolute_positives(similarity_matrix, soft_positives_per_database, k=CORRECT_NUMS+1):
    if soft_positives_per_database == None:
        absolute_positives_per_database = []

        for i in range(similarity_matrix.shape[0]):
            top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]
            absolute_positives = [idx for idx in top_k_indices]
            absolute_positives_per_database.append(absolute_positives)

        return absolute_positives_per_database

    absolute_positives_per_database = []

    for i, soft_positives in enumerate(soft_positives_per_database):
        if len(soft_positives) == 0:
            absolute_positives_per_database.append([])
            continue

        soft_similarities = similarity_matrix[i, soft_positives]

        top_k_indices = np.argsort(soft_similarities)[-k:][::-1]
        absolute_positives = [soft_positives[idx] for idx in top_k_indices]
        absolute_positives_per_database.append(absolute_positives)

    return absolute_positives_per_database

def test(args, eval_ds, model):
    """Compute features of the given dataset and compute the recalls."""
    
    # if args.efficient_ram_testing:
        # return test_efficient_ram_usage(args, eval_ds, model, test_method)
    
    # normal process
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
        database_features_dir = os.path.join(eval_ds.dataset_folder, "database_features.npy")
        queries_features_dir = os.path.join(eval_ds.dataset_folder, "queries_features.npy")

        if os.path.isfile(database_features_dir) == 1:
            database_features = np.load(database_features_dir)
        else: 
            for images, indices in tqdm(database_dataloader, ncols=100):
                features = model(images.to("cuda"))
                features = features.cpu().numpy()
                all_features[indices.numpy(), :] = features
            
            database_features = all_features[:eval_ds.database_num]
            np.save(database_features_dir, database_features)
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_infer_batch_size = args.infer_batch_size
        # queries_infer_batch_size = 1
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        
        if os.path.isfile(queries_features_dir) == 1:
            queries_features = np.load(queries_features_dir)
        else: 
            for inputs, indices in tqdm(queries_dataloader, ncols=100):
                features = model(inputs.to("cuda"))
                features = features.cpu().numpy()
                # if pca is not None:
                #     features = pca.transform(features)
                
                all_features[indices.numpy(), :] = features
            
            queries_features = all_features[eval_ds.database_num:]
            np.save(queries_features_dir, queries_features)
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del all_features

    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    # 1-st end
    if args.dataset_name != "msls":
        soft_positives_per_database = eval_ds.get_positives_database()
    else:
        soft_positives_per_database = None
    similarity_matrix = cosine_similarity(database_features)
    absolute_positives_per_database = get_absolute_positives(similarity_matrix, soft_positives_per_database)
    # save absolute_positives

    # ''' 
    # L2(x - (y1 + y2 + y3) / 3)
    # '''
    rank_n = 10
    new_predictions = []

    for query_idx in range(predictions.shape[0]):
        prediction = predictions[query_idx]
        query_feature = queries_features[query_idx]

        embodied_candidates = [absolute_positives_per_database[pre] for pre in prediction[:rank_n]]

        embodied_features = weighted_feature(query_feature, [database_features[cand] for cand in embodied_candidates], query_idx)
        cosine_similarities = (np.dot(embodied_features, query_feature) / (
                        np.linalg.norm(embodied_features, axis=1) * np.linalg.norm(query_feature)))
        distances = 1-cosine_similarities

        ranked_indices = np.argsort(distances)
        ranked_prediction = prediction[:rank_n][ranked_indices]

        unranked_predictions = prediction[rank_n:]
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