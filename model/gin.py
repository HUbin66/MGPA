import torch
from torch import nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Linear, Sequential
import torch.nn.functional as F

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GIN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.relu = nn.ReLU()

        # 首层
        self.convs.append(
            GINConv(
                Sequential(
                    Linear(in_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU(),
                    Linear(hidden_channels, hidden_channels),
                    nn.BatchNorm1d(hidden_channels),
                    nn.ReLU()
                )
            )
        )
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        nn.BatchNorm1d(hidden_channels),
                        nn.ReLU()
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # 最后一层
        self.convs.append(
            GINConv(
                Sequential(
                    Linear(hidden_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    Linear(out_channels, out_channels),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU()
                )
            )
        )
        self.bns.append(nn.BatchNorm1d(out_channels))

        # 移除图池化层
        # self.lin = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.relu(x)
            x = self.dropout(x)

        # 注释掉图池化
        # x = global_mean_pool(x, batch)

        # 返回节点级别的特征
        return F.log_softmax(x, dim=1)