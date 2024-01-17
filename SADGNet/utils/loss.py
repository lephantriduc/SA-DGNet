# coding=utf-8
import torch
import numpy as np
import torch.nn.functional as F


def hit_loss(y_pred, e, mask1):
    pdf = F.softmax(y_pred[:, :, 0], dim=-1)
    event1 = torch.sign(e)
    tmp1 = event1*torch.log(torch.sum(mask1*pdf, dim=-1)+1e-06)
    tmp2 = (1 - event1)*torch.log((1 - torch.sum(pdf*mask1, dim=-1))+1e-06)
    loss = -torch.mean(tmp1+tmp2)
    assert not torch.isnan(loss)
    return loss


def hit_loss_cr(y_pred, e, mask1):
    loss = 0
    for i in range(2):
        pdf = F.softmax(y_pred[:, :, i], dim=-1)
        ei = torch.where(e == i+1, torch.ones_like(e), torch.zeros_like(e))
        event1 = torch.sign(ei)
        mask1i = mask1[:, i, :]
        tmp1 = event1*torch.log(torch.sum(mask1i*pdf, dim=-1)+1e-06)
        tmp2 = (1 - event1)*torch.log((1 - torch.sum(pdf*mask1i, dim=-1))+1e-06)
        loss = loss + -torch.mean(tmp1+tmp2)
        assert not torch.isnan(loss)
    return loss


def hit_rank_loss(y_pred, k, mask2, t):
    num_event = 1
    t = torch.unsqueeze(t, dim=1)
    k = torch.unsqueeze(k, dim=1)
    eta = []
    for e in range(num_event):
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e + 1).float()
        I_2 = torch.diag_embed(torch.squeeze(I_2))
        tmp_e = F.softmax(y_pred[:, :, e], dim=-1)
        R = torch.mm(tmp_e, mask2.t())
        diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
        R = torch.mm(one_vector, diag_R.t()) - R
        R = R.t()
        T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t, one_vector.t())))
        T = torch.mm(I_2, T)
        temp_eta = torch.mean(T * torch.exp(-R), 1, keepdim=True)
        eta.append(temp_eta)
    eta = torch.stack(eta, dim=1)
    eta = torch.mean(torch.reshape(eta, (-1, num_event)), 1, keepdim=True)
    loss = torch.sum(eta)
    assert not torch.isnan(loss)
    return loss


def hit_rank_loss_cr(y_pred, k, mask2, t):
    num_event = 2
    t = torch.unsqueeze(t, dim=1)
    k = torch.unsqueeze(k, dim=1)
    eta = []
    for e in range(num_event):
        mask2i = mask2[:, e, :]
        one_vector = torch.ones_like(t, dtype=torch.float32)
        I_2 = torch.eq(k, e + 1).float()
        I_2 = torch.diag_embed(torch.squeeze(I_2))
        tmp_e = F.softmax(y_pred[:, :, e], dim=-1)
        R = torch.mm(tmp_e, mask2i.t())
        diag_R = torch.reshape(torch.diagonal(R), (-1, 1))
        R = torch.mm(one_vector, diag_R.t()) - R
        R = R.t()
        T = F.relu(torch.sign(torch.mm(one_vector, t.t()) - torch.mm(t, one_vector.t())))
        T = torch.mm(I_2, T)
        temp_eta = torch.mean(T * torch.exp(-R), 1, keepdim=True)
        eta.append(temp_eta)
    eta = torch.stack(eta, dim=1)
    eta = torch.mean(torch.reshape(eta, (-1, num_event)), 1, keepdim=True)
    loss = torch.sum(eta)
    assert not torch.isnan(loss)
    return loss


def mae_loss(y_pred, t, e, time_index):
    time_index = np.insert(time_index, 0, 0)
    time_index = torch.from_numpy(time_index).float()
    pdf = F.softmax(y_pred[:, :, 0], dim=-1)
    cif1 = torch.cumsum(pdf, dim=1)
    survf = 1 - cif1
    survf = torch.clamp(survf, 0, 1)
    uncensored_index = e == 1
    t_unc = t[uncensored_index]
    if len(t_unc) == 0:
        return 0
    survf_unc = survf[uncensored_index, :]
    time_diff = torch.diff(time_index.reshape(1, -1))
    T = torch.sum(survf_unc* time_diff, dim=1)
    total_mae = torch.abs(T - t_unc)
    return torch.mean(total_mae)


def mae_loss_cr(y_pred, t, e, time_index):
    loss = torch.tensor(0)
    time_index = np.insert(time_index, 0, 0)
    time_index = torch.from_numpy(time_index).float()
    for i in range(2):
        pdf = F.softmax(y_pred[:, :, i], dim=-1)
        cif1 = torch.cumsum(pdf, dim=1)
        survf = 1 - cif1
        survf = torch.clamp(survf, 0, 1)
        uncensored_index = e == (i+1)
        t_unc = t[uncensored_index]
        if len(t_unc) == 0:
            continue
        survf_unc = survf[uncensored_index, :]
        time_diff = torch.diff(time_index.reshape(1, -1))
        T = torch.sum(survf_unc * time_diff, dim=1)
        total_mae = torch.abs(T - t_unc)
        loss = loss + torch.mean(total_mae)
    return loss
