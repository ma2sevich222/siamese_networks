import pandas as pd
import numpy as np
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_train_test_data(train_df, test_df, back_size):
    Close = train_df["Close"].values
    dif = np.diff(Close)
    train_df = train_df[1:-1]
    target = [1 if d > 0 else 0 for d in dif]
    train_df["Target"] = target[1:]
    train_samples = [
        train_df[i - back_size : i]
        for i in range(len(train_df) + 1)
        if i - back_size >= 0
    ]
    train_features = []
    train_labels = []
    scaler = MinMaxScaler()
    for sample in train_samples:
        sample_nrmlzd = scaler.fit_transform(
            sample[["Open", "High", "Low", "Close", "Volume"]].values
        )

        train_features.append(np.array(sample_nrmlzd).reshape(-1, back_size * 5))
        train_labels.append(sample["Target"].iloc[-1])

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    test_samples = [
        test_df[i - back_size : i]
        for i in range(len(test_df) + 1)
        if i - back_size >= 0
    ]
    test_dates = []
    test_features = []

    for sample in test_samples:
        sample_nrmlzd = scaler.fit_transform(
            sample[["Open", "High", "Low", "Close", "Volume"]].values
        )

        test_features.append(np.array(sample_nrmlzd).reshape(-1, back_size * 5))
        test_dates.append(sample["Datetime"].iloc[-1])
        test_features.np.array(test_features)
    return train_features, train_labels, test_features, test_dates


class RBMDataset(Dataset):
    def __init__(self, train_features, train_labels):

        self.x_train = torch.tensor(train_features, dtype=torch.float32)
        self.y_train = torch.tensor(train_labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.x_train[idx], self.y_train[idx]


class RBM(nn.Module):
    def __init__(self, n_vis, n_hin, k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))
        self.k = k

    def sample_from_p(self, p):
        return F.relu(torch.sign(p - Variable(torch.rand(p.size()))))

    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v):
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(self.k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()


"""ФУНКЦИИ ДЕРЕВЬЕВ """


def torch_kron_prod(a, b):
    res = torch.einsum("ij,ik->ijk", [a, b])
    res = torch.reshape(res, [-1, np.prod(res.shape[1:])])
    return res


def torch_bin(x, cut_points, temperature=0.1):
    # x is a N-by-1 matrix (column vector)
    # cut_points is a D-dim vector (D is the number of cut-points)
    # this function produces a N-by-(D+1) matrix, each row has only one element being one and the rest are all zeros
    D = cut_points.shape[0]
    W = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1), [1, -1])
    cut_points, _ = torch.sort(
        cut_points
    )  # make sure cut_points is monotonically increasing
    b = torch.cumsum(torch.cat([torch.zeros([1]), -cut_points], 0), 0)
    h = torch.matmul(x, W) + b
    res = torch.exp(h - torch.max(h))
    res = res / torch.sum(res, dim=-1, keepdim=True)
    return h


def nn_decision_tree(x, cut_points_list, leaf_score, temperature=0.1):
    # cut_points_list contains the cut_points for each dimension of feature
    leaf = reduce(
        torch_kron_prod,
        map(
            lambda z: torch_bin(x[:, z[0] : z[0] + 1], z[1], temperature),
            enumerate(cut_points_list),
        ),
    )
    return torch.matmul(leaf, leaf_score)
