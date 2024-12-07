import os, random
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_metric_learning import losses, miners, distances

from torch.utils.tensorboard import SummaryWriter

from utils import util, parser, commons, test
from models import vgl_network, dinov2_network, er_network
from datasets import gsv_cities, base_dataset, er_dataset
from torch.utils.data.dataset import Subset

from ranking import test_embodied, rank_loss
import faiss
from sklearn.metrics.pairwise import cosine_similarity

cities_name = ["trondheim", "london", "boston", "melbourne", "amsterdam","helsinki",
              "tokyo","toronto","saopaulo","moscow","zurich","paris","bangkok",
              "budapest","austin","berlin","ottawa","phoenix","goa","amman","nairobi","manila"]

torch.backends.cudnn.benchmark = True  # Provides a speedup
#### Initial setup: parser, logging...
args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, "gsv_cities", start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")
logging.info(f"Using {torch.cuda.device_count()} GPUs")

#### Initialize model
model = vgl_network.VGLNet(args)
model = model.to("cuda")

if args.aggregation == "netvlad":
    if args.use_linear:
        args.features_dim = args.clusters * args.linear_dim
        if args.use_cls:
            args.features_dim += 256
    else:
        args.features_dim = args.clusters * dinov2_network.CHANNELS_NUM[args.backbone]
    
if args.resume != None:
    logging.info(f"Resuming model from {args.resume}")
    model = util.resume_model(args, model)

model = torch.nn.DataParallel(model)
model = model.eval()

args.val_positive_dist_threshold = 25
val_ds = base_dataset.BaseDataset(args, "val")
val_database_features, val_queries_features = test.test_feature(args, val_ds, model)
val_similarity_matrix = cosine_similarity(val_database_features)
# val_soft_positives_per_database = val_ds.get_positives_database()
val_absolute_positives_per_database = test_embodied.get_absolute_positives(val_similarity_matrix, None, k = 3)
val_positives_per_query = val_ds.get_positives()
val_recalls, val_recalls_str, val_predictions = test.test(args, val_ds, val_queries_features, val_database_features)
logging.info(f"Recalls on {val_ds}: {val_recalls_str}")
er_val_ds = er_dataset.ERDataset_test(10, val_queries_features, val_database_features, val_absolute_positives_per_database, val_predictions, val_positives_per_query)
val_dl = DataLoader(er_val_ds, batch_size = args.train_batch_size, num_workers=args.num_workers, pin_memory= True)

del val_ds, val_similarity_matrix, val_queries_features, val_database_features, er_val_ds, val_absolute_positives_per_database, val_predictions, val_positives_per_query
torch.cuda.empty_cache()

er_model = er_network.EmbodiedAttention(dim=args.features_dim, num_heads=8)
er_model = er_model.to("cuda")
er_model = torch.nn.DataParallel(er_model)
criterion_binary = torch.nn.BCELoss()
criterion_rank = rank_loss.ListLoss()
optimizer = torch.optim.Adam(er_model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95)

start_epoch_num = 0
best_val_loss = float('inf')
no_improvement_epochs = 0
early_stop_threshold = 5
save_path = "er_model.pth"
alpha = 0.5

for epoch_num in range(start_epoch_num, args.epochs_num):
    er_model.train()
    running_loss_total = 0.0
    val_loss_total = 0.0
    
    for city_name in cities_name:
        # Step 1: feature extraction
        args.val_positive_dist_threshold = 25
        train_ds = base_dataset.BaseDataset(args, "train", city_name)
        train_database_features, train_queries_features = test.test_feature(args, train_ds, model)
        # Step 2: build real training set
        train_similarity_matrix = cosine_similarity(train_database_features)
        # train_soft_positives_per_database = train_ds.get_positives_database()
        train_absolute_positives_per_database = test_embodied.get_absolute_positives(train_similarity_matrix, None, k = 3)
        train_positives_per_query = train_ds.get_positives()
        train_recalls, train_recalls_str, train_predictions = test.test(args, train_ds, train_queries_features, train_database_features)
        logging.info(f"Recalls on {train_ds}: {train_recalls_str}")
        # Step 3: train the Embodied Re-ranking Network
        er_train_ds = er_dataset.ERDataset(10, train_queries_features, train_database_features, train_absolute_positives_per_database, train_predictions, train_positives_per_query)
        train_dl = DataLoader(er_train_ds, batch_size= args.train_batch_size, num_workers=args.num_workers, pin_memory= True)

        progress_bar = tqdm(train_dl, ncols=100, desc=f"Epoch {epoch_num+1}/{args.epochs_num}")
        for batch_idx, (query_feature, neighbors_features, labels, ranks, _) in enumerate(progress_bar):
            query_feature = query_feature.to("cuda")
            neighbors_features = neighbors_features.to("cuda")
            labels = labels.to("cuda")
            ranks = ranks.to("cuda")

            optimizer.zero_grad()
            scores, pre_ranks = er_model(query_feature, neighbors_features)
            scores = scores.squeeze(-1)
            pre_ranks = pre_ranks.squeeze(-1)
            loss_binary = criterion_binary(scores, labels)
            loss_rank = criterion_rank(pre_ranks, ranks)
            loss = alpha * loss_binary + (1 - alpha) * loss_rank

            loss.backward()
            optimizer.step()


            running_loss_total += loss.item()

            progress_bar.set_postfix({
                "City": city_name,
                "Total Loss": f"{running_loss_total:.4f}"
            })

        scheduler.step()

        er_model.eval()
        

        with torch.no_grad():
            for val_query_feature, val_neighbors_features, val_labels, val_ranks, _ in val_dl:
                val_query_feature = val_query_feature.to("cuda")
                val_neighbors_features = val_neighbors_features.to("cuda")
                val_labels = val_labels.to("cuda")
                val_ranks = val_ranks.to("cuda")

                val_scores, val_pre_ranks = er_model(val_query_feature, val_neighbors_features)
                val_scores = val_scores.squeeze(-1)
                val_pre_ranks = val_pre_ranks.squeeze(-1)

                val_loss_binary = criterion_binary(val_scores, val_labels)
                val_loss_rank = criterion_rank(val_pre_ranks, val_ranks)
                val_loss = alpha * val_loss_binary + (1 - alpha) * val_loss_rank
                val_loss_total += val_loss.item()

    print(f"Epoch [{epoch_num + 1}/{args.epochs_num}] - Validation Loss: {val_loss_total:.4f}")
    
    if val_loss_total < best_val_loss:
        best_val_loss = val_loss_total
        no_improvement_epochs = 0
        torch.save(er_model.state_dict(), save_path)
        print(f"Validation loss improved to {best_val_loss:.4f}. Model saved to {save_path}.")
    else:
        no_improvement_epochs += 1
        print(f"No improvement in validation loss for {no_improvement_epochs} epochs.")

    if no_improvement_epochs >= early_stop_threshold:
        print(f"Early stopping triggered after {early_stop_threshold} epochs with no improvement.")
        break

state_dict = torch.load("er_model.pth", map_location="cuda")
er_model.load_state_dict(state_dict)

val_dl = DataLoader(er_val_ds, batch_size = 1, num_workers=args.num_workers, pin_memory= True)
progress_bar = tqdm(val_dl, ncols=100,)
for batch_idx, (query_feature, neighbors_features,_, _, index) in enumerate(progress_bar):
    query_feature = query_feature.to("cuda")
    neighbors_features = neighbors_features.to("cuda")
    scores, pre_ranks = er_model(query_feature, neighbors_features)
    scores = scores.squeeze(-1).cpu().detach().numpy()
    pre_ranks = pre_ranks.squeeze(-1).cpu().detach().numpy()
    reranked_indices = np.argsort(-scores.flatten())
    reranked_indices = np.argsort(pre_ranks.flatten())
    val_predictions[index][:10] = val_predictions[index][:10][reranked_indices]
    
    
recalls = np.zeros(len(args.recall_values))
for query_index, pred in enumerate(val_predictions):
    for i, n in enumerate(args.recall_values):
        if np.any(np.in1d(pred[:n], val_positives_per_query[query_index])):
            recalls[i:] += 1
            break

recalls = recalls / val_ds.queries_num * 100
recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])

logging.info(f"Recalls on {val_ds}: {recalls_str}")
            
            