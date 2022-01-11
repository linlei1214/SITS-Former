from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import BERT, BERTPrediction

torch.manual_seed(0)


class BERTTrainer:

    def __init__(self, bert: BERT, num_features: int,
                 train_loader: DataLoader, valid_loader: DataLoader,
                 lr: float = 1e-3, warmup_epochs: int = 10, decay_gamma: float = 0.99,
                 gradient_clipping_value=5.0, with_cuda: bool = True, cuda_devices=None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTPrediction(bert, num_features).to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.warmup_epochs = warmup_epochs
        self.optim_schedule = lr_scheduler.ExponentialLR(self.optim, gamma=decay_gamma)
        self.gradient_clippling = gradient_clipping_value
        self.criterion = nn.MSELoss(reduction='none')

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Using %d GPUs for model pre-training" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            torch.backends.cudnn.benchmark = True

        self.writer = SummaryWriter()

    def train(self, epoch):
        self.model.train()

        data_iter = tqdm(enumerate(self.train_loader),
                         desc="EP_%s:%d" % ("train", epoch),
                         total=len(self.train_loader),
                         bar_format="{l_bar}{r_bar}")

        train_loss = 0.0
        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            mask_prediction = self.model(data["bert_input"].float(),
                                         data["timestamp"].long(),
                                         data["bert_mask"].long())

            loss = self.criterion(mask_prediction, data["bert_target"].float())
            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clippling)
            self.optim.step()

            train_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": train_loss / (i + 1),
                "loss": loss.item()
            }

            if i % 10 == 0:
                data_iter.write(str(post_fix))

        train_loss = train_loss / len(data_iter)
        self.writer.add_scalar('train_loss', train_loss, global_step=epoch)

        valid_loss = self.validate()
        self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

        if epoch >= self.warmup_epochs:
            self.optim_schedule.step()
        self.writer.add_scalar('cosine_lr_decay', self.optim_schedule.get_lr()[0], global_step=epoch)

        print("EP%d, train_loss=%.5f, validate_loss=%.5f" % (epoch, train_loss, valid_loss))
        return train_loss, valid_loss

    def validate(self):
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        for data in self.valid_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                mask_prediction = self.model(data["bert_input"].float(),
                                             data["timestamp"].long(),
                                             data["bert_mask"].long())

                loss = self.criterion(mask_prediction, data["bert_target"].float())

            mask = data["loss_mask"].unsqueeze(-1)
            loss = (loss * mask.float()).sum() / mask.sum()

            valid_loss += loss.item()
            counter += 1

        valid_loss /= counter
        return valid_loss

    def save(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)

        output_path = os.path.join(path, "checkpoint.tar")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
        }, output_path)

        bert_path = os.path.join(path, "checkpoint.bert.tar")
        torch.save(self.bert.state_dict(), bert_path)
        self.bert.to(self.device)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, path):
        input_path = os.path.join(path, "checkpoint.tar")

        try:
            checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            self.model.train()

            print("Model loaded from:" % input_path)
            return input_path
        except IOError:
            print("Error: parameter file does not exist!")


