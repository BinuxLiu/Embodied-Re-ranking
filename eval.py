import os
import logging
from datetime import datetime
import torch

from utils import parser, commons, util, test
from models import vgl_network, dinov2_network
from datasets import base_dataset

import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from ranking import test_embodied

args = parser.parse_arguments()
start_time = datetime.now()
args.save_dir = os.path.join("logs", args.save_dir, args.backbone + "_" + args.aggregation, args.dataset_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
commons.setup_logging(args.save_dir)
commons.make_deterministic(args.seed)
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

model = vgl_network.VGLNet_Test(args)
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

if args.pca_dim is None:
    pca = None
else:
    full_features_dim = args.features_dim
    args.features_dim = args.pca_dim
    pca = util.compute_pca(args, model, args.pca_dataset_folder, full_features_dim)

test_ds = base_dataset.BaseDataset(args, "test")
logging.info(f"Test set: {test_ds}")

model = model.eval()
with torch.no_grad():
    logging.debug("Extracting database features for evaluation/testing")
    # For database use "hard_resize", although it usually has no effect because database images have same resolution
    database_subset_ds = Subset(test_ds, list(range(test_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=True)
    all_features = np.empty((len(test_ds), args.features_dim), dtype="float32")
    database_features_dir = os.path.join(test_ds.dataset_folder, "database_features.npy")
    queries_features_dir = os.path.join(test_ds.dataset_folder, "queries_features.npy")

    if os.path.isfile(database_features_dir) == 1:
        database_features = np.load(database_features_dir)
    else: 
        for images, indices in tqdm(database_dataloader, ncols=100):
            features = model(images.to("cuda"))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features 
        database_features = all_features[:test_ds.database_num]
        np.save(database_features_dir, database_features)
    
    logging.debug("Extracting queries features for evaluation/testing")
    queries_infer_batch_size = args.infer_batch_size
    # queries_infer_batch_size = 1
    queries_subset_ds = Subset(test_ds, list(range(test_ds.database_num, test_ds.database_num+test_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=queries_infer_batch_size, pin_memory=True)
    
    if os.path.isfile(queries_features_dir) == 1:
        queries_features = np.load(queries_features_dir)
    else: 
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to("cuda"))
            features = features.cpu().numpy()
            if pca is not None:
                features = pca.transform(features)
            all_features[indices.numpy(), :] = features
        
        queries_features = all_features[test_ds.database_num:]
        np.save(queries_features_dir, queries_features)

# Then, we can test difference methods

if args.ranking == "normal":
    recalls, recalls_str = test.test(args, test_ds, queries_features, database_features)
elif args.ranking == "er":
    recalls, recalls_str = test_embodied.test(args, test_ds, queries_features, database_features)
elif args.ranking == "er_net":
    pass

logging.info(f"Recalls on {test_ds}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

