import logging
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from srcs.utils import instantiate


logger = logging.getLogger('evaluate')


@hydra.main(version_base=None,config_path='conf', config_name='evaluate')
def main(config):
    logger.info('Loading checkpoint: {} ...'.format(config.checkpoint))
    checkpoint = torch.load(config.checkpoint)

    loaded_config = OmegaConf.create(checkpoint['config'])

    # setup data_loader instances
    print(config.data_loader)
    data_loader = instantiate(config.data_loader)

    # restore network architecture
    model = instantiate(loaded_config.arch)
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
    logger.info(model)

    # load trained weights
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # instantiate loss and metrics
    criterion = instantiate(loaded_config.loss, is_func=True)
    metrics = [instantiate(met, is_func=True) for met in loaded_config.metrics]

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metrics))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = criterion(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metrics):
                total_metrics[i] += metric(output.cpu(), target.cpu()) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metrics)
    })
    logger.info(log)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
