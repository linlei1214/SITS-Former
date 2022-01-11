import torch
from torch.utils.data import DataLoader
from model import BERT
from trainer import BERTFineTuner
from dataset import FinetuneDataset
import numpy as np
import random
import os
import argparse
import pickle as pkl


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
        help="Path to the labeled dataset.",
    )
    parser.add_argument(
        "--pretrain_path",
        default=None,
        type=str,
        required=False,
        help="The storage path of the pre-trained model parameters.",
    )
    parser.add_argument(
        "--finetune_path",
        default='../checkpoints/finetune',
        type=str,
        required=False,
        help="The output directory where the fine-tuning checkpoints will be written.",
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
        type=int,
        help="List of cuda devices.",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of loader worker processes.",
    )
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
        "--num_classes",
        default=15,
        type=int,
        help="Number of classes.",
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
        default=2e-4,
        type=float,
        help="",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
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
        default=128,
        type=int,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    return parser.parse_args()


if __name__ == "__main__":
    setup_seed(0)
    config = Config()

    train_path = os.path.join(config.dataset_path, 'TRAIN')
    valid_path = os.path.join(config.dataset_path, 'VALIDATE')
    test_path = os.path.join(config.dataset_path, 'TEST')

    print("Loading datasets...")
    train_dataset = FinetuneDataset(train_path, config.num_features, config.patch_size,
                                    config.max_length)
    valid_dataset = FinetuneDataset(valid_path, config.num_features, config.patch_size,
                                    config.max_length)
    test_dataset = FinetuneDataset(test_path, config.num_features, config.patch_size,
                                   config.max_length)
    print("Training samples: %d, validation samples: %d, testing samples: %d" %
          (train_dataset.TS_num, valid_dataset.TS_num, test_dataset.TS_num))

    print("Creating dataloader...")
    train_data_loader = DataLoader(train_dataset, shuffle=True, num_workers=config.num_workers,
                                   batch_size=config.batch_size, drop_last=False)
    valid_data_loader = DataLoader(valid_dataset, shuffle=False, num_workers=config.num_workers,
                                   batch_size=config.batch_size, drop_last=False)
    test_data_loader = DataLoader(test_dataset, shuffle=False, num_workers=config.num_workers,
                                  batch_size=config.batch_size, drop_last=False)

    print("Initialing SITS-Former...")
    bert = BERT(num_features=config.num_features,
                hidden=config.hidden_size,
                n_layers=config.layers,
                attn_heads=config.attn_heads,
                dropout=config.dropout)
    if config.pretrain_path is not None:
        print("Loading pre-trained model parameters...")
        bert_path = os.path.join(config.pretrain_path, 'checkpoint.bert.tar')
        if os.path.exists(bert_path):
            bert.load_state_dict(torch.load(bert_path))
        else:
            print('Cannot find the pre-trained parameter file, please check the path!')

    print("Creating downstream task trainer...")
    trainer = BERTFineTuner(bert, config.num_classes,
                            train_loader=train_data_loader,
                            valid_loader=valid_data_loader,
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                            with_cuda=config.with_cuda,
                            cuda_devices=config.cuda_devices)

    print("Training/Fine-tuning SITS-Former...")
    Best_OA = 0
    Best_Kappa = 0
    Best_F1 = 0
    for epoch in range(config.epochs):
        _, _, valid_oa, valid_kappa, valid_F1score = trainer.train(epoch)
        if valid_F1score >= Best_F1:
            Best_OA = valid_oa
            Best_Kappa = valid_kappa
            Best_F1 = valid_F1score
            trainer.save(epoch, config.finetune_path)
    print('Best performance on the validation set: OA = %.2f%%, Kappa = %.4f, medium_F1score = %.2f%%' %
          (Best_OA, Best_Kappa, Best_F1))

    print("\n")
    print("Testing SITS-Former...")
    trainer.load(config.finetune_path)
    test_OA, test_kappa, test_F1score, confusion_matrix, test_report = trainer.test(test_data_loader)
    print('Best performance on the test set: OA = %.2f%%, Kappa = %.4f, medium_F1score = %.2f%%' %
          (test_OA, test_kappa, test_F1score))
    pkl.dump(confusion_matrix, open(os.path.join(config.finetune_path, 'conf_mat.pkl'), 'wb'))
    print(test_report)

