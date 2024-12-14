# coding=utf-8
import torch
import torch.nn.functional as F
from SADGNet.utils.utils import get_optimizer, DiscreteLabels2, EarlyStopping
from SADGNet.utils.metrics import calculate_c_index_cr, calculate_mae_cr
from SADGNet.utils.logger import my_logger
from SADGNet.utils.data_loader import make_dataloader2
from SADGNet.utils.loss import hit_rank_loss_cr, hit_loss_cr, mae_loss_cr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from .SADGNetSR import SADGNetTorch


class SADGNetCRs(BaseEstimator):
    def __init__(self, learning_rate=1e-3, num_layers=1, hidden_dim=64, dropout=0.0, activation='ReLU',
                 embedding_dim=64, random_seed=2023, epochs=250, batch_size=64, optimizer="Adam", val_size=0.2,
                 time_interval=1, lambda1=1.0, lambda2=1.0, lambda3=0.1, trans_layer=1, alpha=2.0):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.trans_layer = trans_layer
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation = activation
        self.random_seed = random_seed
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.val_size = val_size
        self.time_interval = time_interval
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.fitted = False
        self.model = None
        self.sequence_length = None
        self.time_index = None
        self.time_index_id = None

    def get_torch_model(self, input_dim, num_events):
        return SADGNetTorch(input_dim=input_dim,
                            hidden_dim=self.hidden_dim,
                            embedding_dim=self.embedding_dim,
                            num_layers=self.num_layers,
                            trans_layer=self.trans_layer,
                            dropout=self.dropout,
                            activation=self.activation,
                            num_events=num_events,
                            sequence_length=self.sequence_length,
                            time_index_id=torch.tensor(self.time_index_id),
                            alpha=self.alpha)

    def get_train_val_data(self, x, y):
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=self.val_size, stratify=y[:, 1],
                                                          random_state=self.random_seed)
        return x_train, x_val, y_train, y_val

    def fit(self, x, y):
        my_logger.info(f'hypeparameters: learning_rate {self.learning_rate}, num_layers {self.num_layers}, '
                       f'hidden_dim {self.hidden_dim}, dropout {self.dropout}, activation {self.activation}, '
                       f'batch_size {self.batch_size}, optimizer {self.optimizer}, embedding_dim {self.embedding_dim},'
                       f'time_interval {self.time_interval},lambda1 {self.lambda1}, lambda2 {self.lambda2}, '
                       f'lambda3 {self.lambda3}, trans_layer {self.trans_layer}, alpha {self.alpha}')
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        # data preprocess
        x_train, x_val, y_train_time_event, y_val_time_event = self.get_train_val_data(x, y)
        input_dim = x_train.shape[1]
        dis_label = DiscreteLabels2()
        dis_label.fit(y, time_interval=self.time_interval)
        dis_label.transform(y_train_time_event)
        y_train_mask1 = dis_label.mask1
        y_train_mask2 = dis_label.mask2
        dis_label.transform(y_val_time_event)
        y_val_mask1 = dis_label.mask1
        y_val_mask2 = dis_label.mask2
        t_train = y_train_time_event[:, 0]
        t_val = y_val_time_event[:, 0]
        e_train = y_train_time_event[:, 1]
        e_val = y_val_time_event[:, 1]
        self.time_index = dis_label.time_index
        self.time_index_id = dis_label.time_index_id
        self.sequence_length = len(self.time_index)
        self.model = self.get_torch_model(input_dim, dis_label.max_e)
        optimizer = get_optimizer(self.model, self.learning_rate, self.optimizer)
        dics = []
        dics.append(self.model.state_dict())
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(self.epochs):
            # train
            self.model.train()
            self.fitted = False
            my_logger.info('-------------epoch {}/{}-------------'.format(epoch + 1, self.epochs))
            train_loader = make_dataloader2(x_train, t_train, e_train, y_train_mask1, y_train_mask2,
                                            self.batch_size)
            val_loader = make_dataloader2(x_val, t_val, e_val, y_val_mask1, y_val_mask2, self.batch_size)
            total_train_loss = 0
            for _, (xb, tb, eb, mb1, mb2) in enumerate(train_loader):
                yb_pred = self.model(xb)
                train_loss1 = hit_loss_cr(yb_pred, eb, mb1)
                train_loss2 = hit_rank_loss_cr(yb_pred, eb, mb2, tb)
                train_loss3 = mae_loss_cr(yb_pred, tb, eb, self.time_index)
                train_loss = self.lambda1 * train_loss1 + self.lambda2 * train_loss2 + \
                             self.lambda3 * train_loss3
                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                total_train_loss = total_train_loss + train_loss.item()
            total_train_loss = total_train_loss / len(train_loader)
            # validation
            self.model = self.model.eval()
            self.fitted = True
            train_cindex1, train_cindex2 = self.get_c_index(x_train, y_train_time_event)
            val_cindex1, val_cindex2 = self.get_c_index(x_val, y_val_time_event)
            train_mae1, train_mae2 = self.get_mae(x_train, y_train_time_event)
            val_mae1, val_mae2 = self.get_mae(x_val, y_val_time_event)
            total_val_loss = 0
            with torch.no_grad():
                for _, (val_xb, val_tb, val_eb, val_mb1, val_mb2) in enumerate(val_loader):
                    val_pred = self.model(val_xb)
                    val_loss1 = hit_loss_cr(val_pred, val_eb, val_mb1)
                    val_loss2 = hit_rank_loss_cr(val_pred, val_eb, val_mb2, val_tb)
                    val_loss3 = mae_loss_cr(val_pred, val_tb, val_eb, self.time_index)
                    val_loss = self.lambda1 * val_loss1 + self.lambda2 * val_loss2 + \
                               self.lambda3 * val_loss3
                    total_val_loss = total_val_loss + val_loss.item()
                total_val_loss = total_val_loss / len(val_loader)
            my_logger.info(f'train loss={format(total_train_loss, ".4f")}, '
                           f'val loss={format(total_val_loss, ".4f")}, '
                           f'train_cindex_event1={format(train_cindex1, ".4f")}, train_cindex_event2={format(train_cindex2, ".4f")},'
                           f'val_cindex_event1={format(val_cindex1, ".4f")},val_cindex_event2={format(val_cindex2, ".4f")},'
                           f' train_mae_event1={format(train_mae1, ".4f")},train_mae_event2={format(train_mae2, ".4f")},'
                           f'val_mae_event1={format(val_mae1, ".4f")},val_mae_event2={format(val_mae2, ".4f")}')
            # early stop
            early_stopping(total_val_loss, self.model, dics)
            if early_stopping.early_stop:
                my_logger.info("Early stopping")
                self.model.load_state_dict(dics[0])
                del dics
                break
        self.model = self.model.eval()
        self.fitted = True
        return self

    def score(self, x, y):
        c1, c2 = self.get_c_index(x, y)
        return (c1 + c2) / 2

    def predict_survival(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float()
        if self.fitted:
            with torch.no_grad():
                pred = self.model(x)
                survival_probability = []
                for i in range(2):
                    p_probability = F.softmax(pred[:, :, i], dim=-1).cpu().detach().numpy().T
                    survp = 1 - np.cumsum(p_probability, axis=0)
                    survival_probability.append(np.where(survp < 0, 0, survp))
                return self.time_index, survival_probability

        else:
            raise Exception("The model has not been fitted yet.")

    def get_c_index(self, x, y):
        time_index, probability_array = self.predict_survival(x)
        return calculate_c_index_cr(time_index, probability_array, y)

    def get_mae(self, x, y):
        time_index, probability_array = self.predict_survival(x)
        return calculate_mae_cr(time_index, probability_array, y)

