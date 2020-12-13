import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from models.etm import ETM
from utils.args_parser import parse_args, flatten_cfg, mkdir, save_yaml, newest
from utils.constants import Cte
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-f', '--config_file', default='params/trainer.yaml', type=str)
args = parser.parse_args()

cfg = parse_args(args.config_file)

pl.seed_everything(cfg['seed'])

# %% Load dataset
data_module = None
if cfg['dataset']['name'] == Cte.NG:
    from datasets.news_group import NewsGroupDataModule
    data_module = NewsGroupDataModule(**cfg['dataset']['params'])

assert data_module is not None
cfg['model']['params']['vocab_size'] = data_module.vocab_size

# %% Load model
model = ETM(**cfg['model']['params'])

model.set_optim_params(optim_params=cfg['optimizer'],
                       sched_params=cfg['scheduler'])


# %% Prepare training
save_dir = mkdir(os.path.join(cfg['root_dir'], cfg['dataset']['name'], model.get_model_folder(), str(cfg['seed'])))
# trainer = pl.Trainer(**cfg['model'])
logger = TensorBoardLogger(save_dir=save_dir, name='logs', default_hp_metric=False)
logger.log_hyperparams(flatten_cfg(cfg))

save_dir_ckpt = mkdir(os.path.join(save_dir, 'ckpt'))

checkpoint = ModelCheckpoint(monitor='val_ELBO', mode='max',
                             save_top_k=1, period=5,
                             filename='checkpoint-{epoch:02d}',
                             dirpath=save_dir_ckpt)
early_stopping = EarlyStopping('val_ELBO', mode='max', min_delta=0.0, patience=10)

ckpt_file = newest(save_dir_ckpt)
if ckpt_file is not None:
    print(f'Loading model traning: {ckpt_file}')
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint, early_stopping], resume_from_checkpoint=ckpt_file,
                         **cfg['trainer'])
else:
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint, early_stopping], **cfg['trainer'])


# %% Train

trainer.fit(model, data_module)
save_yaml(model.get_arguments(), file_path=os.path.join(save_dir, 'hparams_model.yaml'))
save_yaml(cfg, file_path=os.path.join(save_dir, 'hparams_full.yaml'))

# %% Testing
out = trainer.test()

model.show_topics(data_module.vocab)
nearest_neighbors = model.nearest_neighbors('world', data_module.vocab)
topic_diversity = model.get_topic_diversity(topk=10)
data_loader = data_module.test_dataloader()

topic_coherence = model.get_topic_coherence(corpus=data_loader.dataset.corpus.numpy())
print(f'Experiment folder: {logger.log_dir}')
for ckpt_file, value in checkpoint.best_k_models.items():
    print(f"{ckpt_file}: {value.item():.2f}")
