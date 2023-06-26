import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
import os
import argparse
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from im2mesh import config, data
# from im2mesh.checkpoints import CheckpointIO
# from im2mesh import onet
# from im2mesh.onet import models, training, generation
import config   ### just need to import config for config.py - if want to take specific method in config then use from config import "name of method"
from checkpoints import CheckpointIO
from onet import generation
from training import training
import models
from torch.utils import data


path="ocnet.yaml"
with open(path, 'r') as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set t0
# t0 = time.time()

# Shorthands
out_dir = "out/onet"
batch_size = 64
backup_every = 100000
#exit_after = args.exit_after  ### can check later if we really need the exit after

model_selection_metric = "iou"
model_selection_sign = 1

# Output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# Dataset
train_dataset = config.get_dataset('train', cfg)
val_dataset = config.get_dataset('val', cfg)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, num_workers=4, shuffle=True,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=10, num_workers=4, shuffle=False,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)


# For visualizations
vis_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=12, shuffle=True,
    collate_fn=collate_remove_none,
    worker_init_fn=worker_init_fn)
data_vis = next(iter(vis_loader))
# Model
model = config.get_model(cfg, device=device, dataset=train_dataset)

# Intialize training
npoints = 1000
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
# trainer = config.get_trainer(model, optimizer, cfg, device=device)
trainer = training.Trainer(                         ## set the trainig for training
    model, optimizer,
    device=device, input_type="img",
    vis_dir="out/img/onet/vis", threshold=0.2,
    eval_sample=False,
)

## all here to load check point and get info from the check point
checkpoint_io = CheckpointIO(out_dir, model=model, optimizer=optimizer)  ##save check poins
try:                                                            ##here to load checkpoint to cotinue to train i guess
    load_dict = checkpoint_io.load('model.pt')         ## so here model saved as .pt not .ckpt
    #load_dict = checkpoint_io.load('model.ckpt')   #so load_dict is a dict for the ckpt file
except FileExistsError:
    load_dict = dict()
epoch_it = load_dict.get('epoch_it', -1)
it = load_dict.get('it', -1)
metric_val_best = load_dict.get(
    'loss_val_best', -model_selection_sign * np.inf)



print('Current best validation metric (%s): %.8f'
      % (model_selection_metric, metric_val_best))   ## model_selection_metric here is IoU metric

# TODO: reintroduce or remove scheduler?
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4000,
#                                       gamma=0.1, last_epoch=epoch_it)
logger = SummaryWriter(os.path.join(out_dir, 'logs'))

# Shorthands
print_every = cfg['training']['print_every']
checkpoint_every = cfg['training']['checkpoint_every']
validate_every = cfg['training']['validate_every']
visualize_every = cfg['training']['visualize_every']

# Print model                                                   ##can remove later- just need to see the number of paras
nparameters = sum(p.numel() for p in model.parameters())
print(model)
print('Total number of parameters: %d' % nparameters)

while True:
    epoch_it += 1
#     scheduler.step()

    for batch in train_loader:
        it += 1
        loss = trainer.train_step(batch)
        logger.add_scalar('train/loss', loss, it)

        # Print output
        if print_every > 0 and (it % print_every) == 0:
            print('[Epoch %02d] it=%03d, loss=%.4f'
                  % (epoch_it, it, loss))

        #Visualize output
        if visualize_every > 0 and (it % visualize_every) == 0:
            print('Visualizing')
            trainer.visualize(data_vis)

        # Save checkpoint
        if (checkpoint_every > 0 and (it % checkpoint_every) == 0):
            print('Saving checkpoint')
            checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)

        # Backup if necessary
        if (backup_every > 0 and (it % backup_every) == 0):
            print('Backup checkpoint')
            checkpoint_io.save('model_%d.pt' % it, epoch_it=epoch_it, it=it,
                               loss_val_best=metric_val_best)
        # Run validation
        if validate_every > 0 and (it % validate_every) == 0:
            eval_dict = trainer.evaluate(val_loader)
            metric_val = eval_dict[model_selection_metric]
            print('Validation metric (%s): %.4f'
                  % (model_selection_metric, metric_val))

            for k, v in eval_dict.items():
                logger.add_scalar('val/%s' % k, v, it)

            if model_selection_sign * (metric_val - metric_val_best) > 0:
                metric_val_best = metric_val
                print('New best model (loss %.4f)' % metric_val_best)
                checkpoint_io.save('model_best.pt', epoch_it=epoch_it, it=it,
                                   loss_val_best=metric_val_best)


        ## this exit can do later if want to
        # # Exit if necessary
        # if exit_after > 0 and (time.time() - t0) >= exit_after:
        #     print('Time limit reached. Exiting.')
        #     checkpoint_io.save('model.pt', epoch_it=epoch_it, it=it,
        #                        loss_val_best=metric_val_best)
        #     exit(3)





def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)

def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)