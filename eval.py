import os
import logging
from datetime import datetime
import torch

from utils import parser, commons, util, test
from models import vgl_network, dinov2_network
from datasets import base_dataset

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
    database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
    database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=True)
    all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")

    for inputs, indices in tqdm(database_dataloader, ncols=100):
        features = model(inputs.to("cuda"))
        features = features.cpu().numpy()
        if pca is not None:
            features = pca.transform(features)
        all_features[indices.numpy(), :] = features        
    # print(model.all_time / eval_ds.database_num)
    
    logging.debug("Extracting queries features for evaluation/testing")
    queries_infer_batch_size = args.infer_batch_size
    # queries_infer_batch_size = 1
    queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
    queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                    batch_size=queries_infer_batch_size, pin_memory=True)
    for inputs, indices in tqdm(queries_dataloader, ncols=100):
        features = model(inputs.to("cuda"))
        features = features.cpu().numpy()
        if pca is not None:
            features = pca.transform(features)
        
        all_features[indices.numpy(), :] = features

queries_features = all_features[eval_ds.database_num:]
database_features = all_features[:eval_ds.database_num]

# Then, we can test difference methods

if args.ranking == "normal":
    recalls, recalls_str = test.test(args, queries_features, database_features)
elif args.ranking == "er":
    pass
elif args.ranking == "er_net":
    pass

logging.info(f"Recalls on {test_ds}: {recalls_str}")
logging.info(f"Finished in {str(datetime.now() - start_time)[:-7]}")

