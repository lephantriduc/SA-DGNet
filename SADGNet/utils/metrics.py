from pycox.evaluation import EvalSurv
import pandas as pd
import numpy as np


def calculate_c_index(time_index, probability_array, y):
    t = y[:, 0]
    e = y[:, 1]
    surv = pd.DataFrame(probability_array, index=time_index)
    ev = EvalSurv(
        surv,
        t,
        e,
        censor_surv="km"
    )
    return ev.concordance_td('antolini')


def calculate_mae(time_index, probability_array, y):
    # probability_array： seq_len*N
    t = y[:, 0]
    e = y[:, 1]
    uncensored_index = e != 0
    t_unc = t[uncensored_index]
    # p_unc: len(uncensored_index)*seq_len
    p_unc = probability_array.T[uncensored_index]
    time_index = np.insert(time_index, 0, 0)
    time_diff = np.diff(time_index.reshape(1, -1))
    T = np.sum(p_unc*time_diff, axis=1)
    total_mae = np.absolute(T-t_unc)
    return np.mean(total_mae)


def calculate_c_index_cr(time_index, probability_array, y):
    t = y[:, 0]
    e = y[:, 1]
    surv1 = pd.DataFrame(probability_array[0], index=time_index)
    surv2 = pd.DataFrame(probability_array[1], index=time_index)
    ev1 = EvalSurv(surv1, t, e == 1, censor_surv='km')
    ev2 = EvalSurv(surv2, t, e == 2, censor_surv='km')
    c_index1 = ev1.concordance_td('antolini')
    c_index2 = ev2.concordance_td('antolini')
    return c_index1, c_index2


def calculate_mae_cr(time_index, probability_array, y):
    mae = []
    time_index = np.insert(time_index, 0, 0)
    for i in range(2):
        pi = probability_array[i].T
        time_index = np.array(time_index)
        times = y[:, 0]
        events = y[:, 1]
        # cause-spesific
        uncensored_index = events == (i+1)
        t_unc = times[uncensored_index]
        p_unc = pi[uncensored_index, :]
        time_diff = np.diff(time_index.reshape(1, -1))
        T = np.sum(p_unc * time_diff, axis=1)
        mae.append(np.mean(np.absolute(T - t_unc)))
    return mae[0], mae[1]
