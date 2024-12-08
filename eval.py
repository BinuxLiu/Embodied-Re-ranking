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

# Then, we can test difference methods

database_features, queries_features = test.test_feature(args, test_ds, model)

if args.ranking == "normal":
    recalls, recalls_str, _ = test.test(args, test_ds, queries_features, database_features)
elif args.ranking == "er":
    recalls, recalls_str = test_embodied.test(args, test_ds, queries_features, database_features)
elif args.ranking == "er_net":
    recalls, recalls_str = test_ernet.test(args, test_ds, queries_features, database_features)

logging.info(f"Recalls on {test_ds}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

