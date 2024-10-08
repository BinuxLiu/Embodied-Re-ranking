import faiss, os
import logging
import numpy as np
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from utils import visualizations

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd
from sklearn.metrics.pairwise import cosine_similarity

CORRECT_NUMS = 4
RERANKS = 20

class embodied_reranking():
    def __init__(self, eval_ds, predictions, database_features):

        self.extractor = SuperPoint(max_num_keypoints=1024).eval().cuda()  # load the extractor
        self.matcher = LightGlue(features='superpoint').eval().cuda()  # load the matcher
        self.queries_num = eval_ds.queries_num
        self.queries_paths = eval_ds.queries_paths
        self.database_paths = eval_ds.database_paths
        self.database_featurs = database_features
        self.predictions = predictions

        if eval_ds.dataset_name != "msls":
            self.soft_positives_per_database = eval_ds.get_positives_database()
        else:
            self.soft_positives_per_database = None
            
        self.similarity_matrix = cosine_similarity(database_features)
        self.absolute_positives_per_database = self.get_absolute_positives(self.similarity_matrix, self.soft_positives_per_database)

    def matching(self, feats0_, pred_path):

        image_pred = load_image(pred_path).cuda()
        feats1 = self.extractor.extract(image_pred, resize = None)
        matches01 = self.matcher({'image0': feats0_, 'image1': feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0_, feats1, matches01]]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        return len(m_kpts0)


    def reranking(self):
        for query in tqdm(range(self.queries_num)):
            query_path = self.queries_paths[query]
            image_query = load_image(query_path).cuda()
            feats0 = self.extractor.extract(image_query, resize=None)  # auto-resize the image, disable with resize=None

            match_scores = []
            for i in range(RERANKS):
                recall_image = self.predictions[query, i]
                voting_images = self.absolute_positives_per_database[recall_image]

                score = 0
                for voting_image in voting_images:
                    score += self.matching(feats0, self.database_paths[voting_image])

                match_scores.append((score, recall_image))

            match_scores.sort(reverse=True, key=lambda x: x[0])
            self.predictions[query, :RERANKS] = [x[1] for x in match_scores[:RERANKS]]

        return self.predictions
    
    @staticmethod
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



def test_efficient_ram_usage(args, eval_ds, model):
    model = model.eval()

    distances = np.empty([eval_ds.queries_num, eval_ds.database_num], dtype=np.float32)
    
    with torch.inference_mode():
        queries_features = np.ones((eval_ds.queries_num, args.features_dim), dtype="float32")
        queries_infer_batch_size = 1
        # queries_infer_batch_size = args.infer_batch_size
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=True)
        for images, indices in tqdm(queries_dataloader, ncols=100):
            features = model(images.to("cuda"))
            queries_features[indices.numpy()-eval_ds.database_num, :] = features.cpu().numpy()

        queries_features = torch.tensor(queries_features).type(torch.float32).cuda()

        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=True)
        database_descriptors_dir = os.path.join(eval_ds.dataset_folder, f"database_{args.aggregation}.npy")
        if os.path.isfile(database_descriptors_dir) == 1:
            database_descriptors = np.load(database_descriptors_dir)
            total_size = database_descriptors.shape[0]
            distances = np.empty((queries_features.shape[0], total_size), dtype=np.float32)

            for batch_start in tqdm(range(0, total_size, 512)):
                batch_end = min(batch_start + 521, total_size)
                batch = database_descriptors[batch_start:batch_end]
                batch = torch.from_numpy(batch).to('cuda')

                for index, pred_feature in enumerate(batch):
                    global_index = batch_start + index
                    distances[:, global_index] = ((queries_features - pred_feature) ** 2).sum(1).cpu().numpy()

                del batch
            del queries_features, database_descriptors
        else: 
            all_descriptors = np.empty((len(eval_ds), args.features_dim), dtype="float32")
            for images, indices in tqdm(database_dataloader, ncols=100):
                descriptors = model(images.to("cuda"))
                descriptors = descriptors.cpu().numpy()
                all_descriptors[indices.numpy(), :] = descriptors
            database_descriptors = all_descriptors[:eval_ds.database_num]
            np.save(database_descriptors_dir, database_descriptors)
            database_descriptors = torch.from_numpy(database_descriptors)
            database_descriptors = database_descriptors.to('cuda')

            for index, pred_feature in enumerate(database_descriptors):
                    distances[:, index] = ((queries_features - pred_feature) ** 2).sum(1).cpu().numpy()
            del features, queries_features, pred_feature, database_descriptors

    predictions = distances.argsort(axis=1)[:, :max(args.recall_values)]
    del distances

    positives_per_query = eval_ds.get_positives()

    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break

    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

    # Save visualizations of predictions
    if args.num_preds_to_save != 0:
        logging.info("Saving final predictions")
        # For each query save num_preds_to_save predictions
        visualizations.save_preds(predictions[:, :args.num_preds_to_save], eval_ds,
                                args.save_dir, args.save_only_wrong_preds)
        
    return recalls, recalls_str

def test(args, eval_ds, model , pca = None):
    """Compute features of the given dataset and compute the recalls."""
    
    if args.efficient_ram_testing:
        return test_efficient_ram_usage(args, eval_ds, model)
    
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

    # rerank = embodied_reranking(eval_ds, predictions, database_features)
    # predictions = rerank.reranking()

    del database_features
    
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