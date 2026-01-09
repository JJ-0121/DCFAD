import torch
import numpy as np
import os

from torch_frft.dfrft_module import dfrft
from tqdm import tqdm
from model.DCFAD import DCFAD
from data_factory.data_loader import get_loader_segment
import torch.nn as nn
from utils.affiliation.generics import convert_vector_to_events
from utils.affiliation.metrics import pr_from_events
import time

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class OneEarlyStopping:
    def __init__(self, patience=3, verbose=False, dataset_name='', delta=0, warmup=1):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name
        self.warmup = warmup
        self.epoch = 0
    def __call__(self, val_loss, model, path):
        self.epoch += 1
        if self.epoch <= self.warmup:
            if self.verbose:
                print(f"Warm-up epoch, skipping early stopping check.")
            if self.best_score is None:
                self.save_checkpoint(val_loss, model, path)
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss:{val_loss:.6f}.  Saving model ...')

        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + f'_checkpoint.pth'))
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='thre',
                                              dataset=self.dataset)
        self.criterion = nn.MSELoss()
        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.build_model()

    def build_model(self):
        self.model = DCFAD(c_in=self.input_c, c_out=self.output_c,revin=self.revin, d_model=self.d_model, e_layers=self.e_layers,win_size=self.win_size,dropout=self.dr, device=self.device,f_tr=self.f_tr,conv_kernel_list=self.c_list).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        with torch.no_grad():
            for i, (input_data, _) in enumerate(vali_loader):
                input = input_data.float().to(self.device)
                z= self.model(input)
                recon_loss_tf =  self.criterion(z, input)
                loss_1.append((recon_loss_tf).item())
        return np.average(loss_1)

    def train(self):
        print("======================TRAIN MODE======================")
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        train_steps = len(self.train_loader)
        early_stopping = OneEarlyStopping(
            patience=3,
            verbose=True,
            dataset_name=self.dataset
        )

        for epoch in range(self.num_epochs):
            start_epoch_time = time.time()
            loss1_list = []

            self.model.train()
            with tqdm(total=train_steps) as pbar:
                for i, (input_data, _) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()

                    input = input_data.float().to(self.device)

                    # forward
                    z = self.model(input)
                    recon_loss_tf = self.criterion(z, input)
                    loss1_list.append(recon_loss_tf.item())
                    loss1 = recon_loss_tf
                    loss1.backward()
                    self.optimizer.step()
                    pbar.update(1)


            train_loss = np.average(loss1_list)
            vali_loss1 = self.vali(self.vali_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss1: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            early_stopping(vali_loss1, self.model, path)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.dataset) + '_checkpoint.pth')
            )
        )
        self.model.eval()


        print("======================TEST MODE======================")

        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            z = self.model(input)
            loss1 = torch.mean(criterion(z, input), dim=-1)
            cri =   loss1
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            z= self.model(input)
            loss1=torch.mean(criterion(z, input), dim=-1)
            cri = loss1
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            z = self.model(input)
            loss1 = torch.mean(criterion(z, input), dim=-1)
            cri = loss1
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        def getAffiliationMetrics(label, pred):
            events_pred = convert_vector_to_events(pred)
            events_label = convert_vector_to_events(label)
            Trange = (0, len(pred))
            result = pr_from_events(events_pred, events_label, Trange)
            P = result['precision']
            R = result['recall']
            F = 2 * P * R / (P + R)
            return P, R, F
        precision, recall, f1_score = getAffiliationMetrics(gt.copy(), pred.copy())
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(gt, test_energy)
        print(
            "affiliation_precision : {:0.3f}, affiliation_recall : {:0.3f}, affiliation_f : {:0.3f}, AUC-ROC : {:0.3f}".format(
                precision,
                recall, f1_score,auc))
        return precision, recall, f1_score,auc
