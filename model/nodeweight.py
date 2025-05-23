from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from Utils import com_top_K_patter_x, sup_pattern

from concurrent.futures import ThreadPoolExecutor

from test import read_file_and_count_attributes


def cosine_similarity(vector_a, vector_b):
    """
    计算两个向量的余弦相似度
    :param vector_a: 第一个向量，numpy数组
    :param vector_b: 第二个向量，numpy数组
    :return: 余弦相似度值，范围在 [-1, 1]
    """
    # 计算向量的点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算向量的模
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 计算余弦相似度
    if norm_a == 0 or norm_b == 0:
        # 如果任一向量为零向量，相似度为0
        return 0
    else:
        return dot_product / (norm_a * norm_b)




# def normalize(weights):
#     # sum=0
#     # for i in range(len(weights)):
#     #     sum+=weights[i]
#     # mf=sum/len(weights)
#     # sum=0
#     # for i in range(len(weights)):
#     #     sum+=(abs(weights[i]-mf))
#     # sf=sum/len(weights)
#     # for i in range(len(weights)):
#     #     weights[i]=(weights[i]-mf)/sf
#     min_val = np.min(weights)
#     max_val = np.max(weights)
#     # return weights
#     return (weights - min_val) / (max_val - min_val)

def get_node_weight(x,file_name,k,v_num,nfeatuer,gnn_model):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    x_tensor = torch.tensor(x, dtype=torch.float32)
    x_tensor = x_tensor.to('cpu')
    # x_tensor = x_tensor.clone().detach().float()
    # x_tensor=x_tensor.tolist()
    x_fre_pattern = com_top_K_patter_x(file_name, nfeatuer, gnn_model)
    # x_fre_pattern = x_fre_pattern.to(device)
    # x_fre_pattern = x_fre_pattern.tolist()
    # x_fre_pattern = torch.tensor(x_fre_pattern)


    # 计算欧氏距离
    supmin = sup_pattern(file_name, k)
    Wnode = np.zeros(v_num)

    for j in range(len(x_fre_pattern)):
        x_fre_pattern[j] = torch.mean(x_fre_pattern[j], axis=0)

    # x_fre_pattern = torch.stack(x_fre_pattern)
    # x_fre_pattern = torch.sum(x_fre_pattern, dim=0)
    # supmin =np.flip(supmin)

    # sup=read_file_and_count_attributes("Data/pattern_graph/old_graph/mico/200.lg")
    # for i in range(len(x)):
    #     max_idx = torch.argmax(x[i])  # 获取第 i 行最大值的索引
    #     Wnode[i] = sup.get(max_idx.item(), 0)  # 根据索引从 sup 中取出对应的值

    # Wnode = np.where(torch.isnan(Wnode), 0, Wnode)



    for i in range(len(x)):
        if i%1000==0 and i !=0: print(i)
        # min_value = x_tensor[i].min()
        # max_value = x_tensor[i].max()
        # normalized_array = (x_tensor[i] - min_value) / (max_value - min_value)
        for j in range(len(x_fre_pattern)):
            Wnode[i] +=torch.dot(x_tensor[i],x_fre_pattern[j])*supmin[j]
            # Wnode[i] += (1 / distance) * (supmin[j])
            # if res > distance:
            #     res = distance
            #     cnt = j


    # Wnode=normalize(Wnode)
    min = Wnode.min()
    min=abs(min)
    for i in range(len(x)):
         Wnode[i]+=(min+1e-6)
         # Wnode[i] =0
    wz=0
    max=0
    for i in range(len(Wnode)):
        if max<Wnode[i]:
            max=Wnode[i]
            wz=i
    print(wz)
    # 31745
    # 31745
    #31745
    #59518GAT
    # 97613SAGE
    #31112SAGE
    # 多头GAT：60860
    return Wnode

    # x_tensor = torch.tensor(x, dtype=torch.float64)
    # x_fre_pattern = com_top_K_patter_x(file_name,nfeatuer,gnn_model)
    # for i in range(len(x_fre_pattern)):  # 遍历第一维
    #     for j in range(len(x_fre_pattern[i])):  # 遍历第二维
    #         print(x_fre_pattern[j],end=" ")  # 现在 x_fre_pattern[i, j] 是单个元素
    #     print()
    #
    # # 计算欧氏距离
    # supmin = sup_pattern(file_name, k)
    # Wnode = np.zeros(v_num)
    # max=-1
    # wz=-1
    # torch.set_printoptions(sci_mode=False)
    # for j in range(len(x_fre_pattern)):
    #     x_fre_pattern[j] = torch.mean(x_fre_pattern[j], axis=0)
    # # temp=np.zeros(5000)
    # max=0
    #
    # for i in range(len(x)):
    #     print(str(i)+":",end=" ")
    #     if(i%1000==0 and i!=0):
    #         #  Wnode = normalize(Wnode)
    #         print(i)
    #     for j in range(len(x_fre_pattern)):
    #         # distance = torch.sqrt(torch.sum((x_tensor[i] - x_fre_pattern[j]) ** 2))
    #         distance=cosine_similarity(x_tensor[i].detach().numpy(),x_fre_pattern[j].detach().numpy())+1
    #         # hfunc(x_fre_pattern[j].detach().numpy(),x_tensor[i].detach().numpy())
    #         # if  distance<=  2.71828: distance = torch.tensor(2.7)
    #         # Wnode[i] += ((1 / distance) * (supmin[j]))
    #         Wnode[i] += ( distance) * (supmin[j])
    #         # temp[i]=Wnode[i]
    #         # if Wnode[i]>max: max=i
    #
    #         # print(distance.item(),end=" ")
    #         # Wnode[i] += torch.log(distance) * (supmin[j])
    #
    #
    # Wnode=normalize(Wnode)
    # for i in range(16):
    #     if max<Wnode[i]:
    #         max=Wnode[i]
    #         wz=i
    # res[wz]+=1
    # return Wnode
