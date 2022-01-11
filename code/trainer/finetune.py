import os
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import BERT, BERTClassification
from .focal_loss import FocalLoss
# from .metric import Average_Accuracy, Kappa_Coefficient
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, f1_score


class BERTFineTuner:
    def __init__(self, bert: BERT, num_classes: int,
                 train_loader: DataLoader, valid_loader: DataLoader,
                 criterion='CrossEntropyLoss', lr: float = 1e-3, weight_decay=0,
                 with_cuda: bool = True, cuda_devices=None):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.bert = bert
        self.model = BERTClassification(bert, num_classes)
        self.num_classes = num_classes

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optim = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if criterion == 'FocalLoss':
            self.criterion = FocalLoss(gamma=1)
        else:
            self.criterion = nn.CrossEntropyLoss()

        if with_cuda and torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("Using %d GPUs for model pre-training" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model, device_ids=cuda_devices)
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            torch.backends.cudnn.benchmark = True

        number_parameters = sum([p.nelement() for p in self.model.parameters()]) / 1000000
        print("Total Parameters: %.2f M" % number_parameters)

    def train(self, epoch):
        self.model.train()

        train_loss = 0.0
        counter = 0
        for data in self.train_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            predict = self.model(data["bert_input"].float(),
                                 data["timestamp"].long(),
                                 data["bert_mask"].long())

            loss = self.criterion(predict, data["class_label"].squeeze().long())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss += loss.item()

            counter += 1

        train_loss /= counter

        valid_loss, valid_OA, valid_kappa, valid_F1score = self.validate()
        print("EP%d, Valid Accuracy: OA=%.2f%%, medium_F1_score=%.2f%%" % (epoch, valid_OA, valid_F1score))

        return train_loss, valid_loss, valid_OA, valid_kappa, valid_F1score

    def validate(self):
        self.model.eval()

        valid_loss = 0.0
        counter = 0
        total_correct = 0
        total_element = 0
        y_pred = []
        y_true = []
        for data in self.valid_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():

                y_p = self.model(data["bert_input"].float(),
                                 data["timestamp"].long(),
                                 data["bert_mask"].long())

                y = data["class_label"].view(-1)
                loss = self.criterion(y_p, y.long())

            valid_loss += loss.item()

            y_true.extend(list(map(int, y.cpu())))
            y_p = y_p.argmax(dim=-1)
            y_pred.extend(list(map(int, y_p.cpu())))

            # compute OA
            correct = (y == y_p).sum()
            total_correct += correct
            total_element += y.numel()

            counter += 1

        valid_loss /= counter
        valid_OA = total_correct * 100.0 / total_element
        valid_kappa = cohen_kappa_score(y_true, y_pred, labels=list(range(self.num_classes)))
        valid_F1score = f1_score(y_true, y_pred, average='macro', labels=list(range(self.num_classes))) * 100.0

        return valid_loss, valid_OA, valid_kappa, valid_F1score

    def test(self, data_loader):
        self.model.eval()

        total_correct = 0
        total_element = 0
        y_pred = []
        y_true = []
        for data in data_loader:
            data = {key: value.to(self.device) for key, value in data.items()}

            with torch.no_grad():
                y_p = self.model(data["bert_input"].float(),
                                 data["timestamp"].long(),
                                 data["bert_mask"].long())

                y = data["class_label"].view(-1)

            y_true.extend(list(map(int, y.cpu())))
            y_p = y_p.argmax(dim=-1)
            y_pred.extend(list(map(int, y_p.cpu())))

            # compute OA
            correct = (y == y_p).sum()
            total_correct += correct
            total_element += y.numel()

        test_OA = total_correct * 100.0 / total_element
        test_kappa = cohen_kappa_score(y_true, y_pred)
        test_F1score = f1_score(y_true, y_pred, average='macro', labels=list(range(self.num_classes))) * 100.0
        test_conf = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes))) * 100.0
        test_report = classification_report(y_true, y_pred, labels=list(range(self.num_classes)))

        return test_OA, test_kappa, test_F1score, test_conf, test_report

    def save(self, epoch, path):
        if not os.path.exists(path):
            os.makedirs(path)

        output_path = os.path.join(path, "checkpoint.tar")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, path):
        input_path = os.path.join(path, "checkpoint.tar")

        try:
            checkpoint = torch.load(input_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch = checkpoint['epoch']
            self.model.train()

            print("EP:%d Model loaded from:" % epoch, input_path)
            return input_path
        except IOError:
            print("Error: parameter file does not exist!")



