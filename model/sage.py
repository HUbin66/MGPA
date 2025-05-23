# 导包
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch中的函数模块，通常用于激活函数、损失函数等操作
from torch_geometric.nn import GCNConv, SAGEConv  # 从PyTorch几何库中导入图卷积网络层（GCNConv）

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)