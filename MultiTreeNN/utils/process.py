import torch
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from objects import Metrics, InputDataset, Stat, Stats
import log as logger


class Train(object):
    def __init__(self, step, epochs, verbose=True):
        self.epochs = epochs
        self.step = step
        self.history = History()
        self.verbose = verbose

    def __call__(self, train_loader_step, val_loader_step=None, early_stopping=None):
        for epoch in range(self.epochs):
            self.step.train()
            train_stats = train_loader_step(self.step)
            self.history(train_stats, epoch + 1)

            if val_loader_step is not None:
                with torch.no_grad():
                    self.step.eval()
                    val_stats = val_loader_step(self.step)
                    self.history(val_stats, epoch + 1)

                print(self.history)

                if early_stopping is not None:
                    valid_loss = val_stats.loss()
                    # early_stopping needs the validation loss to check if it has decreased,
                    # and if it has, it will make a checkpoint of the current model
                    if early_stopping(valid_loss):
                        self.history.log()
                        return
            else:
                print(self.history)
        self.history.log()


def predict(step, test_loader_step):
    print(f"Testing")
    with torch.no_grad():
        step.eval()
        stats = test_loader_step(step)
        metrics = Metrics(stats.outs(), stats.labels())
        print(metrics)
        metrics.log()
    return metrics()["Accuracy"]


class History:
    def __init__(self):
        self.history = {}
        self.epoch = 0
        self.timer = time.time()

    def __call__(self, stats, epoch):
        self.epoch = epoch

        if epoch in self.history:
            self.history[epoch].append(stats)
        else:
            self.history[epoch] = [stats]

    def __str__(self):
        epoch = f"\nEpoch {self.epoch};"
        stats = ' - '.join([f"{res}" for res in self.current()])
        timer = f"Time: {(time.time() - self.timer)}"

        return f"{epoch} - {stats} - {timer}"

    def current(self):
        return self.history[self.epoch]

    def log(self):
        msg = f"(Epoch: {self.epoch}) {' - '.join([f'({res})' for res in self.current()])}"
        logger.log_info("history", msg)


# Author: https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, model, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model = model

    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0

        return self.early_stop

    def save_checkpoint(self, val_loss):
        """
            Saves model when validation loss decrease.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.model.save()
        self.val_loss_min = val_loss


def train_val_test_split(data_frame: pd.DataFrame, shuffle=True):
    print("Splitting Dataset")

    false = data_frame[data_frame.target == 0]
    true = data_frame[data_frame.target == 1]

    train_false, test_false = train_test_split(false, test_size=0.2, shuffle=shuffle)
    test_false, val_false = train_test_split(test_false, test_size=0.5, shuffle=shuffle)
    train_true, test_true = train_test_split(true, test_size=0.2, shuffle=shuffle)
    test_true, val_true = train_test_split(test_true, test_size=0.5, shuffle=shuffle)

    train = train_false.append(train_true)
    val = val_false.append(val_true)
    test = test_false.append(test_true)

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return InputDataset(train), InputDataset(test), InputDataset(val)


class LoaderStep:
    def __init__(self, name, data_loader, device):
        self.name = name
        self.loader = data_loader
        self.size = len(data_loader)
        self.device = device

    def __call__(self, step):
        self.stats = Stats(self.name)

        for i, batch in enumerate(self.loader):
            batch.to(self.device)
            stat: Stat = step(i, batch, batch.y)
            self.stats(stat)

        return self.stats