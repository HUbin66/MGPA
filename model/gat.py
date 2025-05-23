import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv
from torch.nn import Linear

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads, dropout):
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.head_channels = hidden_channels // num_heads

        self.convs = torch.nn.ModuleList(
            [GATConv(in_channels, self.head_channels, dropout=dropout) for _ in range(num_heads)]
        )
        self.conv_last = GATConv(self.head_channels * num_heads, out_channels, dropout=dropout)

        # 添加残差连接所需的线性层
        self.lin_skip = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # 原始输入存储用于残差连接
        x_skip = self.lin_skip(x)  # 输入层到输出层的残差连接

        # 多头注意力层
        outputs = [conv(x, edge_index) for conv in self.convs]
        x = torch.cat(outputs, dim=-1)  # 拼接所有头的输出

        # 最后一个注意力层
        x = self.conv_last(x, edge_index)

        # 添加残差连接
        x = x + x_skip  # 隐藏层与输入层的残差连接

        return F.log_softmax(x, dim=1)

