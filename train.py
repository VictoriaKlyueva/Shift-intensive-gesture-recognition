import numpy as np
import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from pathlib import Path
from srcs.trainer import Trainer
from srcs.utils import instantiate, get_logger, is_master

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def train_worker(config):
    if is_master():
        print(config)
    logger = get_logger('train')
    # setup data_loader instances
    data_loader, valid_data_loader = instantiate(config.data_loader)

    # build model. print it's structure and # trainable params.
    model = instantiate(config.arch)
    # добавила код ниже
    model.classifier = nn.Sequential(
        nn.Linear(in_features=model.classifier[1].in_features, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=4096, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=4096, out_features=25, bias=True)
    )
    model.features[0] = nn.Conv2d(1, 32, 3, 3)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    logger.info(model)
    logger.info(f'Trainable parameters: {sum([p.numel() for p in trainable_params])}')

    # get function handles of loss and metrics
    criterion = instantiate(config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in config['metrics']]

    # build optimizer, learning rate scheduler.
    optimizer = instantiate(config.optimizer, model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


def init_worker(working_dir, config):
    # initialize training config
    config = OmegaConf.create(config)
    config.cwd = working_dir
    # prevent access to non-existing keys
    OmegaConf.set_struct(config, True)
    # start training processes
    train_worker(config)


@hydra.main(version_base=None, config_path='conf/', config_name='train')
def main(config):
    n_gpu = torch.cuda.device_count()
    if config.gpu:
        config['n_gpu'] = n_gpu
    else:
        config['n_gpu'] = None

    working_dir = str(Path.cwd().relative_to(hydra.utils.get_original_cwd()))
    if config.resume is not None:
        config.resume = hydra.utils.to_absolute_path(config.resume)
    config = OmegaConf.to_yaml(config, resolve=True)
    init_worker(working_dir, config)

if __name__ == '__main__':
    main()