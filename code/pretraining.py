import torch
from dataset.dataset_wrapper import DataSetWrapper
from model import BERT
from trainer import BERTTrainer
import numpy as np
import random
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--dataset_path",
        default=None,
        type=str,
        required=True,
        help="Path to the unlabeled dataset.",
    )
    parser.add_argument(
        "--pretrain_path",
        default='../checkpoints/pretrain',
        type=str,
        required=False,
        help="The output directory where the pre-training checkpoints will be written.",
    )
    parser.add_argument(
        "--with_cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether cuda is available.",
    )
    parser.add_argument(
        "--cuda_devices",
        default=None,
        type=list,
        help="List of cuda devices.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of loader worker processes.",
    )
    parser.add_argument(
        "--valid_rate",
        default=0.03,
        type=float,
        help="Proportion of samples used for validation")
    parser.add_argument(
        "--max_length",
        default=75,
        type=int,
        help="The maximum length of input time series. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--patch_size",
        default=5,
        type=int,
        help="Size of the input patches.",
    )
    parser.add_argument(
        "--num_features",
        default=10,
        type=int,
        help="The dimensionality of satellite observations.",
    )
    parser.add_argument(
        "--mask_rate",
        default=0.3,
        type=float,
        help="The fraction of timesteps in a time series that will be masked out.",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="Number of hidden neurons of the Transformer network.",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="Number of layers of the Transformer network.",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="Number of attention heads of the Transformer network.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        help="",
    )
    parser.add_argument(
        "--epochs",
        default=200,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
        type=int,
        help="",
    )
    parser.add_argument(
        "--decay_gamma",
        default=0.99,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--gradient_clipping",
        default=5.0,
        type=float,
        help="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(0)
    config = Config()

    print("Loading datasets...")
    dataset = DataSetWrapper(data_path=config.dataset_path,
                             batch_size=config.batch_size,
                             valid_size=config.valid_rate,
                             num_features=config.num_features,
                             patch_size=config.patch_size,
                             max_length=config.max_length,
                             mask_rate=config.mask_rate,
                             num_workers=config.num_workers)

    train_loader, valid_loader = dataset.get_data_loaders()

    print("Initialing SITS-Former...")
    bert = BERT(num_features=config.num_features,
                hidden=config.hidden_size,
                n_layers=config.layers,
                attn_heads=config.attn_heads,
                dropout=config.dropout)

    trainer = BERTTrainer(bert, config.num_features,
                          train_loader=train_loader,
                          valid_loader=valid_loader,
                          lr=config.learning_rate,
                          warmup_epochs=config.warmup_epochs,
                          decay_gamma=config.decay_gamma,
                          gradient_clipping_value=config.gradient_clipping,
                          with_cuda=config.with_cuda,
                          cuda_devices=config.cuda_devices)

    print("Pre-training SITS-Former...")
    mini_loss = np.Inf
    for epoch in range(config.epochs):
        train_loss, valida_loss = trainer.train(epoch)
        if mini_loss > valida_loss:
            mini_loss = valida_loss
            trainer.save(epoch, config.pretrain_path)

