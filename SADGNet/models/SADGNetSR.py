# coding=utf-8
import torch
from torch import nn
import torch.nn.functional as F
from SADGNet.utils.utils import act_str2obj, get_optimizer, DiscreteLabels1, EarlyStopping
from SADGNet.utils.metrics import calculate_c_index, calculate_mae
from SADGNet.utils.logger import my_logger
from SADGNet.utils.data_loader import make_dataloader1
from SADGNet.utils.loss import hit_rank_loss, hit_loss, mae_loss
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator


class DeepGatedNeuralNetwork(nn.Module):
    def __init__(self, feature_dim, time_embedding_dim, hidden_dim, num_layers, activation, dropout):
        super(DeepGatedNeuralNetwork, self).__init__()
        self.activation = activation
        self.dropout_rate = dropout
        self.f_layers = self.mlp(input_dim=feature_dim, hidden_dim=hidden_dim, output_dim=feature_dim,
                                 num_layers=num_layers)
        self.t_layers = self.mlp(input_dim=time_embedding_dim, hidden_dim=hidden_dim, output_dim=feature_dim,
                                 num_layers=num_layers)
        self.sigmoid = nn.Sigmoid()

    def mlp(self, input_dim, hidden_dim, output_dim, num_layers):
        mlp_list = []
        pre_dim = input_dim
        for _ in range(num_layers - 1):
            mlp_list.append(nn.Linear(in_features=pre_dim, out_features=hidden_dim))
            mlp_list.append(act_str2obj(self.activation))
            mlp_list.append(nn.LayerNorm(hidden_dim))
            mlp_list.append(nn.Dropout(self.dropout_rate))
            pre_dim = hidden_dim
        mlp_list.append(nn.Linear(in_features=pre_dim, out_features=output_dim))
        return nn.Sequential(*mlp_list)

    def forward(self, feature, time_embedding):
        f_out = self.f_layers(feature)
        t_out = self.t_layers(time_embedding)
        gate_out = self.sigmoid(t_out)
        cross_out = torch.cat((gate_out * f_out + feature, t_out), dim=-1)
        return cross_out, gate_out


class ScaledDotProductSelfAttention(nn.Module):
    def __init__(self, dq, dk, dv):
        super(ScaledDotProductSelfAttention, self).__init__()
        self.W_q = nn.Linear(dq, dq)
        self.W_k = nn.Linear(dk, dk)
        self.W_v = nn.Linear(dv, dv)
        self._norm_fact = 1 / np.sqrt(dk)

    def forward(self, query, key, value):
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        K_T = K.permute(0, 2, 1)
        attn = nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self._norm_fact)
        output = torch.bmm(attn, V)
        return output, attn


class MultiScaleTimeAwareSelfAttention(nn.Module):
    def __init__(self, dv):
        super(MultiScaleTimeAwareSelfAttention, self).__init__()
        self.w_s1 = nn.Linear(dv, dv // 3)
        self.w_s2 = nn.Linear(dv, dv // 3)
        self.w_s3 = nn.Linear(dv, dv - (dv // 3) * 2)

    def forward(self, x, alpha=5.0):
        v1, v2, v3 = self.w_s1(x), self.w_s2(x), self.w_s3(x)
        out1 = torch.bmm(self.get_local_attn(x, 1.0 / alpha), v1)
        out2 = torch.bmm(self.get_local_attn(x, 1.0), v2)
        out3 = torch.bmm(self.get_local_attn(x, alpha), v3)
        output = torch.cat((out1, out2, out3), dim=-1)
        return output

    def get_local_attn(self, x, alpha=5.0):
        bs, seq_len, _ = x.shape
        t = torch.arange(1, seq_len + 1, dtype=torch.float32).reshape(1, -1)
        S = torch.tile(t, dims=(seq_len, 1))
        S_transpose = torch.transpose(S, dim0=0, dim1=1)
        p_attn = F.softmax(-torch.abs(S - S_transpose) / alpha, dim=-1)
        p_attn = torch.tile(p_attn, dims=(bs, 1, 1))
        return p_attn


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha, drop_out):
        super(SelfAttention, self).__init__()
        self.alpha = alpha
        self.half_dim = input_dim // 2
        self.global_attn = ScaledDotProductSelfAttention(dq=self.half_dim, dk=self.half_dim, dv=self.half_dim)
        self.local_attn = MultiScaleTimeAwareSelfAttention(dv=self.half_dim)
        self.linear1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=input_dim)
        self.relu = nn.ReLU()
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
        attn_x1, g_attn = self.global_attn(x[:, :, :self.half_dim], x[:, :, :self.half_dim], x[:, :, :self.half_dim])
        attn_x2 = self.local_attn(x[:, :, self.half_dim:], alpha=self.alpha)
        x = self.norm1(x + torch.cat((attn_x1, attn_x2), dim=-1))
        x = self.norm2(self.linear2(self.drop_out(self.relu(self.linear1(x)))) + x)
        return x


class SADGNetTorch(nn.Module):
    def __init__(self, input_dim, embedding_dim=64, hidden_dim=256, num_layers=1, dropout=0.1,
                 activation='ReLU', num_events=1, sequence_length=355, time_index_id=None, trans_layer=1, alpha=2.0):
        super(SADGNetTorch, self).__init__()
        self.alpha = alpha
        self.sequence_length = sequence_length
        self.time_index_id = time_index_id
        self.activation = activation
        self.dropout_rate = dropout
        self.gate = None
        self.embedding_table = nn.Embedding(len(time_index_id), embedding_dim)
        self.dgnn = DeepGatedNeuralNetwork(feature_dim=input_dim, time_embedding_dim=embedding_dim,
                                           hidden_dim=hidden_dim, activation=activation,
                                           dropout=dropout, num_layers=num_layers)
        self.sa = self.self_attention(input_dim=input_dim * 4, hidden_dim=hidden_dim, num_layers=trans_layer)
        self.proj_layer = nn.Linear(input_dim * 4, num_events)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.tile(x, dims=(1, self.sequence_length, 1))
        x, self.gate = self.dgnn(x, self.get_time_embedding(x))
        x = self.sa(torch.cat((x, x), dim=-1))
        logits = self.proj_layer(x)
        return logits

    def self_attention(self, input_dim, hidden_dim, num_layers):
        sa_list = []
        for _ in range(num_layers):
            sa_list.append(SelfAttention(input_dim=input_dim, hidden_dim=hidden_dim, alpha=self.alpha,
                                         drop_out=self.dropout_rate))
        return nn.Sequential(*sa_list)

    def get_time_embedding(self, x):
        bs, sequence_length, feature_dim = x.shape
        bs_time_index = self.time_index_id.unsqueeze(0)
        bs_time_index = torch.tile(bs_time_index, dims=(bs, 1))
        time_embedding = self.embedding_table(bs_time_index)
        return time_embedding


class SADGNetSR(BaseEstimator):
    def __init__(self, learning_rate=1e-3, num_layers=1, hidden_dim=64, dropout=0.0, activation='ReLU',
                 embedding_dim=64, random_seed=2023, epochs=250, batch_size=64, optimizer="Adam", val_size=0.2,
                 time_interval=1, lambda1=1.0, lambda2=1.0, lambda3=0.01, trans_layer=1, alpha=2.0):
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
        dis_label = DiscreteLabels1()
        dis_label.fit(y, time_interval=self.time_interval)
        y_train = dis_label.transform(y_train_time_event)
        y_train_mask1 = dis_label.mask1
        y_train_mask2 = dis_label.mask2
        y_val = dis_label.transform(y_val_time_event)
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
            train_loader = make_dataloader1(x_train, y_train, t_train, e_train, y_train_mask1, y_train_mask2,
                                            self.batch_size)
            val_loader = make_dataloader1(x_val, y_val, t_val, e_val, y_val_mask1, y_val_mask2, self.batch_size)
            total_train_loss = 0
            for _, (xb, yb, tb, eb, mb1, mb2) in enumerate(train_loader):
                yb_pred = self.model(xb)
                train_loss1 = hit_loss(yb_pred, eb, mb1)
                train_loss2 = hit_rank_loss(yb_pred, eb, mb2, tb)
                train_loss3 = mae_loss(yb_pred, tb, eb, self.time_index)
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

            train_cindex = self.get_c_index(x_train, y_train_time_event)
            val_cindex = self.get_c_index(x_val, y_val_time_event)
            train_mae = self.get_mae(x_train, y_train_time_event)
            val_mae = self.get_mae(x_val, y_val_time_event)
            total_val_loss = 0
            with torch.no_grad():
                for _, (val_xb, val_yb, val_tb, val_eb, val_mb1, val_mb2) in enumerate(val_loader):
                    val_pred = self.model(val_xb)
                    val_loss1 = hit_loss(val_pred, val_eb, val_mb1)
                    val_loss2 = hit_rank_loss(val_pred, val_eb, val_mb2, val_tb)
                    val_loss3 = mae_loss(val_pred, val_tb, val_eb, self.time_index)
                    val_loss = self.lambda1 * val_loss1 + self.lambda2 * val_loss2 + \
                               self.lambda3 * val_loss3
                    total_val_loss = total_val_loss + val_loss.item()
                    total_val_loss = total_val_loss + val_loss.item()
                total_val_loss = total_val_loss / len(val_loader)
            my_logger.info(f'train loss={format(total_train_loss, ".4f")}, '
                           f'val loss={format(total_val_loss, ".4f")}, '
                           f'train_cindex={format(train_cindex, ".4f")} '
                           f'train_mae={format(train_mae, ".4f")}, val_cindex={format(val_cindex, ".4f")},'
                           f' val_mae={format(val_mae, ".4f")}')
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
        return self.get_c_index(x, y)

    def predict_survival(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).float()
        if self.fitted:
            with torch.no_grad():
                pred = self.model(x)
                p_probability = F.softmax(pred[:, :, 0], dim=-1).cpu().detach().numpy().T
                survival_probability = 1 - np.cumsum(p_probability, axis=0)
                survival_probability = np.where(survival_probability < 0, 0, survival_probability)
                return self.time_index, survival_probability

        else:
            raise Exception("The model has not been fitted yet.")

    def get_c_index(self, x, y):
        time_index, probability_array = self.predict_survival(x)
        return calculate_c_index(time_index, probability_array, y)

    def get_mae(self, x, y):
        time_index, probability_array = self.predict_survival(x)
        return calculate_mae(time_index, probability_array, y)


