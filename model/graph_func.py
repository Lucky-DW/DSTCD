import numpy as np
from numpy import argsort
from sklearn.metrics.pairwise import pairwise_distances
import torch


def pad_or_truncate_tensor(tensor, target_size=1024):
    # 获取当前张量的形状
    current_shape = []
    current_size = tensor.shape[0]
    current_shape.append(current_size)
    padding_size = target_size - current_size
    padding = torch.zeros(padding_size, tensor.shape[1], dtype=tensor.dtype, device=tensor.device)
    out_tensor = torch.cat([tensor, padding], dim=0)
    return out_tensor



def pr_distance(node):
    euc_dis = torch.cdist(node, node, p=2)
    a = 1 - (torch.sqrt(torch.sum(euc_dis)) - torch.mean(euc_dis)) / (
                torch.sqrt(torch.sum(euc_dis)) + torch.mean(euc_dis))
    gaus_dis  = torch.exp(- euc_dis * euc_dis / a)
    return gaus_dis

def graph_construct(objects):
    adjlist = []
    obj_nums = len(objects)
    for i in range(0, obj_nums):
        sub_object = objects[i]
        adj_mat = pr_distance(sub_object)
        norm_adj_mat= normalize_adj(adj_mat)
        adjlist.append([adj_mat, norm_adj_mat])
    return adjlist

def find_sim_node(node):
    size = node.shape[0]
    ####计算欧式距离
    # diff = pairwise_distances(node)
    diff = torch.cdist(node, node,p=2)
    node_1=[]
    for i in range(size):
        a = diff[i,:]
        sortedDistIndex = torch.argsort(a)
        # sortedDistIndex1=sortedDistIndex[::k]
        node_re = node[sortedDistIndex]
        node_1.append(node_re)
    node = torch.stack(node_1)
    node_re = torch.mean(node,axis=0)
    return node_re

# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     d_inv_sqrt = np.power(np.array(adj.sum(1)), -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     return adj.dot(np.diag(d_inv_sqrt)).transpose().dot(np.diag(d_inv_sqrt))

def normalize_adj(adj):
    adj = adj.float()  # 确保操作在浮点数上
    # 计算每行的和
    row_sum = torch.sum(adj, dim=1)
    # 计算倒数平方根
    d_inv_sqrt = row_sum.pow(-0.5)
    # 将无穷大值替换为 0
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    # 构建对角矩阵
    d_inv_sqrt_diag = torch.diag(d_inv_sqrt)
    # 计算最终的结果
    normalized_adj = adj @ d_inv_sqrt_diag @ d_inv_sqrt_diag.T

    return normalized_adj