import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hid_channels, dropout_prob):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hid_channels)
        self.conv2 = GCNConv(hid_channels, num_node_features)
        self.dropout_prob = dropout_prob  # dropout 概率

    def forward(self, x, edge_index):
        # 第一层卷积 + ReLU 激活函数 + dropout
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)

        # 使用 log_softmax 作为输出
        return F.log_softmax(x, dim=1)