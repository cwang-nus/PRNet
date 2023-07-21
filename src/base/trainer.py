import logging
import os
import time
from typing import Optional, List, Union

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam

from src.utils.logging import get_logger
from src.utils.metrics import get_all_metrics
from src.utils import loss

class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            data,
            dataset,
            model_name,
            exp_name,
            ext_flag,
            base_lr: float,
            lr_decay_ratio,
            loss_fn: str,
            log_dir: str,
            n_exp: int,
            save_iter: int = 300,
            clip_grad_value: Optional[float] = None,
            max_epochs: Optional[int] = 1000,
            patience: Optional[int] = 1000,
            device: Optional[Union[torch.device, str]] = None,
    ):
        super().__init__()

        self._logger = get_logger(
            log_dir, __name__, 'info_{}.log'.format(n_exp), level=logging.INFO)
        self._device = device
        self._model = model
        self._ext_flag = ext_flag
        self.model.to(self._device)
        self._logger.info("perform {} on dataset: {} for {}".format(model_name, dataset, exp_name))
        self._logger.info("the number of parameters: {}".format(
            self.model.param_num()))
        if loss_fn == 'l1':
            self._loss_fn = loss.MAELoss()
        elif loss_fn == 'l2':
            self._loss_fn = nn.MSELoss()
        elif loss_fn == 'maesmape':
            self._loss_fn = loss.MAESMAPE_Loss()
        elif loss_fn == 'smoothl1':
            self._loss_fn = loss.SmoothL1Loss()

        self._loss_fn.to(self._device)
        self._base_lr = base_lr
        self._optimizer = Adam(self.model.parameters(), base_lr)

        if lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            self._lr_scheduler = MultiStepLR(self.optimizer,
                                             [max_epochs*0.2, max_epochs*0.4, max_epochs*0.6],
                                             gamma=lr_decay_ratio)
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._save_iter = save_iter
        self._save_path = log_dir
        self._n_exp = n_exp
        self._data = data

    @property
    def model(self):
        return self._model

    @property
    def supports(self):
        return self._supports

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def device(self):
        return self._device

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _inverse_transform(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [self.data['scaler'].inverse_transform(tensor) for tensor in tensors]
        else:
            return self.data['scaler'].inverse_transform(tensors)

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp):
        if not os.path.exists(save_path): os.makedirs(save_path)
        # filename = 'epoch_{}.pt'.format(epoch)
        filename = 'final_model_{}.pt'.format(n_exp)
        torch.save(self.model.state_dict(), os.path.join(save_path, filename))
        return True

    def load_model(self, epoch, save_path, n_exp):
        filename = 'final_model_{}.pt'.format(n_exp)
        self.model.load_state_dict(torch.load(
            os.path.join(save_path, filename)))
        return True

    def early_stop(self, epoch, best_loss):
        self.logger.info(
            'Early stop at epoch {}, loss = {:.6f}'.format(epoch, best_loss))

    def _calculate_supports(self, adj_mat, filter_type):
        return None

    def train_batch(self, X, gt, ext=None):
        self.optimizer.zero_grad()
        pred = self.model(X, ext)

        pred, gt = self._inverse_transform([pred, gt])

        loss = self.loss_fn(pred, gt)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self._clip_grad_value)
        self.optimizer.step()
        return loss.item()

    def train(self):
        self.logger.info("start training !!!!!")

        # training phase
        iter = 0
        val_losses = [np.inf]

        saved_epoch = -1
        for epoch in range(self._max_epochs):

            if epoch - saved_epoch > self._patience:
                # run test when early stop
                self.test(True, True)
                self.early_stop(epoch, min(val_losses))
                break

            self.model.train()
            train_losses = []

            start_time = time.time()
            for i, data in enumerate(self.data['train_loader']):
                if self._ext_flag:
                    XC, XP, XT, Y, YP, YT, gt, ext = self._check_device(data)
                    batch_loss = self.train_batch(
                        (XC, XP, XT, YP, YT), gt, ext)
                else:
                    XC, XP, XT, Y, YP, YT, gt = self._check_device(data)
                    batch_loss = self.train_batch(
                        (XC, XP, XT, YP, YT), gt)

                train_losses.append(batch_loss)
                iter += 1

                if iter % self._save_iter == 0:
                    message = "[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                                self._max_epochs,
                                                                                i,
                                                                                len(
                                                                                    self.data['train_loader']),
                                                                                batch_loss)
                    self._logger.info(message)

            end_time = time.time()
            self.logger.info("epoch complete")

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
                new_lr = self.lr_scheduler.get_lr()[0]
            else:
                new_lr = self._base_lr

            val_loss = self.evaluate()

            message = 'Epoch [{}/{}] ({}) avg_train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, ' \
                      '{:.1f}s'.format(epoch,
                                       self._max_epochs,
                                       iter,
                                       np.mean(train_losses),
                                       val_loss,
                                       new_lr,
                                       (end_time - start_time))
            self._logger.info(message)

            if val_loss < np.min(val_losses):
                self.logger.info("Validating!!!")
                model_file_name = self.save_model(
                    epoch, self._save_path, self._n_exp)
                self._logger.info(
                    'Val loss decrease from {:.4f} to {:.4f}, '
                    'saving to {}'.format(np.min(val_losses), val_loss, model_file_name))
                val_losses.append(val_loss)
                saved_epoch = epoch

            if epoch + 1 == self._max_epochs:
                self.test(True, True)

    def evaluate(self):
        labels = []
        preds = []
        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data['val_loader']):
                if self._ext_flag:
                    XC, XP, XT, Y, YP, YT, _, ext = self._check_device(data)
                    _, pred, Y, YT = self.eval_batch((XC, XP, XT, YP, YT), Y, YT, ext)
                else:
                    XC, XP, XT, Y, YP, YT, _ = self._check_device(data)
                    _, pred, Y, YT = self.eval_batch((XC, XP, XT, YP, YT), Y, YT)

                pred = torch.mean(YT+pred, dim=1)
                pred[pred < 0] = 0

                labels.append(Y.cpu())
                preds.append(pred.cpu())

        labels = torch.cat(labels, dim=0)
        preds = torch.cat(preds, dim=0)

        if type(self.loss_fn) == nn.MSELoss:
            loss = np.sqrt(self.loss_fn(preds, labels).item())
        else:
            loss = self.loss_fn(preds, labels).item()
        return loss

    def eval_batch(self, X, Y, YT, ext=None):
        pred = self.model(X, ext)

        XC, pred, Y, YT = self._inverse_transform([X[0], pred, Y, YT])

        return XC, pred, Y, YT

    def test(self, load_flag, save_flag=False, mode='test'):
        if load_flag:
            self.load_model(-1, self.save_path, self._n_exp)

        labels, preds, inputs = [], [], []

        with torch.no_grad():
            self.model.eval()
            for _, data in enumerate(self.data[mode + '_loader']):

                if self._ext_flag:
                    XC, XP, XT, Y, YP, YT, gt, ext = self._check_device(data)

                    XC, pred, Y, YT = self.eval_batch((XC, XP, XT, YP, YT), Y, YT, ext)
                else:
                    XC, XP, XT, Y, YP, YT, _ = self._check_device(data)
                    XC, pred, Y, YT = self.eval_batch((XC, XP, XT, YP, YT), Y, YT)

                pred = torch.mean(YT+pred, dim=1)
                pred[pred < 0] = 0

                labels.append(Y.cpu())
                preds.append(pred.cpu())
                inputs.append(XC.cpu())

        labels = torch.cat(labels, dim=0).detach().numpy()
        preds = torch.cat(preds, dim=0).detach().numpy()

        metrics = get_all_metrics(preds, labels)
        log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}, Test SMAPE: {:.4f}'.format(metrics[0], metrics[1], metrics[2], metrics[3])
        self._logger.info(log)

        if save_flag:
            inputs = torch.cat(inputs, dim=0).detach().numpy()
            np.save(os.path.join(self.save_path, 'preds'), preds)
            np.save(os.path.join(self.save_path, 'labels'), labels)
            np.save(os.path.join(self.save_path, 'inputs'), inputs)

        return metrics

