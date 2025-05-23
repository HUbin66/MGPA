import csv
from datetime import datetime, time

import networkx as nx
import numpy
import numpy as np
import re

import torch

from motif.get_motif import get_motif
import numpy as np
#
def read_graph_data(file_path):
    js = -1
    graph_data = []
    temp = []
    with open(file_path, 'r') as file:
        for line in file:
            if js==-1:
                js+=1
                continue

            if ":" in line:
                graph_data.append(temp)
                temp = []
                js += 1
                continue
            temp.append(line.strip())
    graph_data.append(temp)



    return graph_data

def pipei(file_name,js,k,l):
    cc='Data/fuzhu_result/EWS/'+file_name+'/'+str(js+1)+"/"+str(k)+'/'+file_name+'_'+str(l)+'.lg'
    graph_b = read_graph_data('Data/fuzhu_result/EWS/now1/'+"/"+str(k)+'/'+file_name+'/'+str(js+1)+"/"+str(k)+'/'+file_name+'_'+str(l)+'.lg')
    graph_a = read_graph_data("Data/pattern_graph/old_graph/"+file_name+'/'+str(k)+".lg")
    count = 0
    # for i in range(len(graph_a)):
    #     for j in range(len(graph_b)):
    #         graph_b[j][-1] = list(graph_b[j][-1])
    #         graph_b[j][-1][-1]='0'
    #         graph_b[j][-1] = ''.join(graph_b[j][-1])
    #         if(graph_a[i]==graph_b[j]):
    #             count+=1
    #
    # return count
    cn=0
    # print(str(l)+":")

    for i in range(len(graph_a)):
        flag = 0
        for j in range(len(graph_b)):
            graph_b[j][-1] = list(graph_b[j][-1])
            a = list(graph_b[j][-1])
            graph_b[j][-1][-1] = '0'
            b = graph_b[j][-1][-1]
            graph_b[j][-1] = ''.join(graph_b[j][-1])
            if (graph_a[i] == graph_b[j]):
                count += 1
                # flag=1
        # if flag==0:
        #     print(str(graph_a[i]))
        # if (i+1)%20==0 :
            # print(count-cn)
            # cn=count
    return count

def xx():

    with open('Data/amazon.txt', 'r') as file:
        data = file.readlines()


    unique_numbers = set()


    res=0;
    for i,line in enumerate(data) :
        if(i==0):
            i+=1
            continue

        parts = line.strip().split()
        if len(parts) >= 3 and parts[0]=='v':
            if(res<int(parts[2])): res=int(parts[2])
            number = int(parts[2])
            unique_numbers.add(number)

    print(f"不同数字的数量: {len(unique_numbers)}")


def get():
    import json

    with open('data.txt', 'r', encoding='utf-8') as file:

        data = json.load(file)
    return data

def storage(data):
    import json

    json_str = json.dumps(data, indent=4)


    with open('data.txt', 'w', encoding='utf-8') as f:
        f.write(json_str)

def result():
    start_time = datetime.now()
    # print()
    file_name_dict = ["dblp", "patent", "mico", "twitch",'twitter']
    dblp = [{200: [0, 0, 0,0, 0, 0]}for _ in range(10)]
    patent = [{200: [0, 0, 0,0, 0, 0]}for _ in range(10)]
    mico = [{200: [0, 0, 0,0, 0, 0]}for _ in range(10)]
    twitch = [{200: [0, 0, 0,0, 0, 0]}for _ in range(10)]
    twitter = [{200: [0, 0, 0, 0, 0, 0]} for _ in range(10)]
    top_k = [200]
    fuzhu = [0.05,0.1,0.15000000000000002,0.2]
    res = [{"dblp": dblp}, {"patent": patent}, {"mico": mico}, {"twitch": twitch},{"twitter": twitter}]
    for i in range(0, 1):
        for file_name in file_name_dict:
            for j in top_k:
                for ll in range(0, 4):
                    l = fuzhu[ll]
                    if (file_name == "dblp"):
                        dblp[i][j][ll] = pipei(file_name, i, j, l)
                    if (file_name == "mico"):
                        mico[i][j][ll] = pipei(file_name, i, j, l)
                    if (file_name == "patent"):
                        patent[i][j][ll] = pipei(file_name, i, j, l)
                    if (file_name == "twitch"):
                        twitch[i][j][ll] = pipei(file_name, i, j, l)

    storage(res)
    # print(dblp)
    return res


def result1(data_name,file_num,K_num):

    path=''
    start_time = datetime.now()
    # print()
    file_name_dict = [f"{data_name}"]
    # file_name_dict = ["dblp"]
    dataset=[{K_num: [0, 0, 0,0, 0, 0]}for _ in range(file_num)]
    top_k = [K_num]
    fuzhu = [0.05,0.1,0.15000000000000002,0.2]
    # fuzhu = [0.05, 0.100, 0.150, 0.200]
    name=data_name
    res=[{f"{data_name}": dataset}]
    # res = [{"dblp": dblp}]
    # res = [{"twitch": twitch}]
    for i in range(0,file_num):
        for file_name in file_name_dict:
            for j in top_k:
                for ll in range(0, 4):
                    l = fuzhu[ll]
                    dataset[i][j][ll] = pipei(file_name, i, j, l)

                    # if (file_name == "twitch"):
                    #     twitch[i][j][ll] = pipei(file_name, i, j, l)
                    # if (file_name == "mico"):
                    #     mico[i][j][ll] = pipei(file_name, i, j, l)
                    # if (file_name == "dblp"):
                    #     dblp[i][j][ll] = pipei(file_name, i, j, l)
                    # if (file_name == "patent"):
                    #     patent[i][j][ll] = pipei(file_name, i, j, l)

    storage(res)
    return res


def averages(res):
    file_name_dict = ["dblp", "patent", "mico", "twitch",'twitter']
    dblp_ava = [{200: [0, 0, 0,0, 0, 0]}]
    patent_ava =[{200: [0, 0, 0,0, 0, 0]}]
    mico_ava = [{200: [0, 0, 0,0, 0, 0]}]
    twitch_ava = [{200: [0, 0, 0,0, 0, 0]}]
    twitter_ava = [{200: [0, 0, 0, 0, 0, 0]}]
    top_k = [200]
    # fuzhu = [0.3, 0.4, 0.5]#l
    res_ava = [{"dblp_ava": dblp_ava}, {"patent_ava": patent_ava}, {"mico_ava": mico_ava}, {"twitch_ava": twitch_ava},{"twitter_ava": twitter_ava}]
    for ff,file_name in enumerate(file_name_dict) :
        for kk,k in enumerate(top_k):
            for l in range(0, 4):
                # for j in range(0, 3):
                    sum = 0
                    for i in range(0, 1):
                        # b=res
                        # a=res[0]['dblp'][0]['300'][0]
                        a=res[ff][file_name][i][str(k)][l]
                        sum+=float(a)

                    # c=a = res_ava[ff][file_name+'_ava']
                    # a = res_ava[ff][file_name+'_ava'][0]
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum/(1*500), ".2f")
                    # res_ava[ff][file_name+'_ava'][str(k)][j]  =sum/10

    print(res_ava)

def averages1(res,data_name,file_num,K_num):
    # file_name_dict = ["twitch"]

    file_name_dict = [data_name]
    file_ava=[{K_num: [0, 0, 0,0,0,0]}]
    top_k = [K_num]
    name=data_name+'_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]
    for ff,file_name in enumerate(file_name_dict) :
        for kk,k in enumerate(top_k):
            for i in range(0, file_num):
                for l in range(0, 6):
                    sum = 0
                    a=res[ff][file_name][i][str(k)][l]
                    sum+=float(a)
                    res_ava[ff][f"{file_name}_ava"][0][k][l]=sum/(K_num)
                res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum/(K_num), ".2f")
                print(res_ava)

    print("--------------------")
    file_name_dict = [data_name]
    file_ava = [{K_num: [0, 0, 0, 0, 0, 0]}]
    top_k = [K_num]
    name = data_name + '_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]
    for ff, file_name in enumerate(file_name_dict):
        for kk, k in enumerate(top_k):
            for l in range(0, 6):
                sum = 0
                for i in range(0, file_num):
                    a = res[ff][file_name][i][str(k)][l]
                    sum += float(a)
                    # res_ava[ff][f"{file_name}_ava"][0][k][l] = sum / (500)
                res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum / (K_num*file_num), ".2f")
        print(res_ava)

def rename():
    import os


    top_k = [200]
    # file_name_dict = ["dblp", "patent", "mico", "twitch"]
    file_name_dict = ["twitter","dblp","patent","twitch","mico"]
    # directory = 'Data/SamplingData/EWS/log'
    directory = 'Data/fuzhu_result/EWS/twitter'
    # directory = 'Data/fuzhu_result/pinjun'
    for i in range(1, 6):
        for file_name in file_name_dict:
            for j in top_k:

                path = os.path.join(directory, str(i), file_name, str(j))

                try:
                    files = os.listdir(path)
                    for file in files:
                        if "0.15000000000000002" in file:
                            new_file_name = file.replace("0.15000000000000002", "0.15")

                            old_file_path = os.path.join(path, file)
                            new_file_path = os.path.join(path, new_file_name)

                            os.rename(old_file_path, new_file_path)
                            print(f"Renamed '{old_file_path}' to '{new_file_path}'")
                except FileNotFoundError:
                    print(f"Directory '{path}' not found.")





def pipei1():
    graph_b = read_graph_data('Data/fuzhu_result/test1')
    graph_a = read_graph_data('Data/fuzhu_result/old/SS/now1/twitter/1/500/twitter-0.200.lg')
    # Data/pattern_graph/old_graph/twitter/500.lg
    # Data/fuzhu_result/test1
    # Data/fuzhu_result/1pattern2.txt
    # aa="Data/fuzhu_result/EWS/twitter/5/200/twitter_0.2.lg"
    # bb="Data/fuzhu_result/RE/200/twitter-0.200.lg"
    # graph_b = read_graph_data(
    #     'Data/fuzhu_result/1/dblp/500/dblp_0.1.lg')

    count = 0
    cn=0
    for i in range(len(graph_a)):
        for j in range(len(graph_b)):
            graph_b[j][-1] = list(graph_b[j][-1])
            graph_b[j][-1][-1] = '0'
            graph_b[j][-1] = ''.join(graph_b[j][-1])
            a=graph_a[i]
            b=graph_b[j]
            if (graph_a[i] == graph_b[j]):
                # print(str(graph_a[i])+"   "+str(graph_b[j]))
                count += 1
        if ((i + 1) % 100 == 0):
            print(count-cn)
            # cn=count
            # count = 0


    return count
    # return format(count / , ".2f")


def result_sj():
    start_time = datetime.now()
    # print()
    file_name_dict = ["dblp", "patent", "mico", "twitch", 'twitter']
    dblp = [{500: [0, 0, 0, 0, 0, 0]}]
    patent = [{500: [0, 0, 0, 0, 0, 0]}]
    mico = [{500: [0, 0, 0, 0, 0, 0]} ]
    twitch = [{500: [0, 0, 0, 0, 0, 0]} ]
    twitter = [{500: [0, 0, 0, 0, 0, 0]} ]
    top_k = [500]
    fuzhu = [0.05, 0.1, 0.15, 0.2]
    res = [{"dblp": dblp}, {"patent": patent}, {"mico": mico}, {"twitch": twitch}, {"twitter": twitter}]
    for i in range(0, 1):
        for file_name in file_name_dict:
            for j in top_k:
                for ll in range(0, 4):
                    l = fuzhu[ll]
                    if (file_name == "dblp"):
                        dblp[i][j][ll] = pipei_sj(file_name, i, j, l)/500
                    if (file_name == "mico"):
                        mico[i][j][ll] = pipei_sj(file_name, i, j, l)/500
                    if (file_name == "patent"):
                        patent[i][j][ll] = pipei_sj(file_name, i, j, l)/500
                    if (file_name == "twitch"):
                        twitch[i][j][ll] = pipei_sj(file_name, i, j, l)/500
                    if (file_name == "twitter"):
                        twitter[i][j][ll] = pipei_sj(file_name, i, j, l)/500

    storage(res)
    print(res)
    return res

def pipei_sj(file_name,js,k,l):
    l = f"{l:.3f}"
    graph_a = read_graph_data('Data/pattern_graph/old_graph/' + file_name + '/' + str(k) + '.lg')
    graph_b = read_graph_data(
        'Data/fuzhu_result/sj/' + file_name + '/' + str(js+1 ) + '/' + str(k) + '/' + file_name + '-' + str(
            l) + '.lg')
    count = 0
    for i in range(len(graph_a)):
        for j in range(len(graph_b)):
            graph_b[j][-1] = list(graph_b[j][-1])
            graph_b[j][-1][-1] = '0'
            graph_b[j][-1] = ''.join(graph_b[j][-1])
            if (graph_a[i] == graph_b[j]):
                # print(str(graph_a[i])+"   "+str(graph_b[j]))
                count += 1


    return count
    # return  format(count/k, ".2f")


def averages_sj(res):
    file_name_dict = ["dblp", "patent", "mico", "twitch",'twitter']
    dblp_ava = [{500: [0, 0, 0,0, 0, 0]}]
    patent_ava =[{500: [0, 0, 0,0, 0, 0]}]
    mico_ava = [{500: [0, 0, 0,0, 0, 0]}]
    twitch_ava = [{500: [0, 0, 0,0, 0, 0]}]
    twitter_ava = [{500: [0, 0, 0, 0, 0, 0]}]
    top_k = [500]
    # fuzhu = [0.3, 0.4, 0.5]#l
    res_ava = [{"dblp_ava": dblp_ava}, {"patent_ava": patent_ava}, {"mico_ava": mico_ava}, {"twitch_ava": twitch_ava},{"twitter_ava": twitter_ava}]
    for ff,file_name in enumerate(file_name_dict) :
        for kk,k in enumerate(top_k):
            for l in range(0, 4):
                # for j in range(0, 3):
                    sum = 0
                    for i in range(0, 1):
                        # b=res
                        # a=res[0]['dblp'][0]['300'][0]
                        a=res[ff][file_name][i][str(k)][l]
                        sum+=float(a)

                    # c=a = res_ava[ff][file_name+'_ava']
                    # a = res_ava[ff][file_name+'_ava'][0]
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum/(1*500), ".2f")
                    # res_ava[ff][file_name+'_ava'][str(k)][j]  =sum/10

    print(res_ava)

def MNI():

    input_file = "Data/Motif/500/dblp.txt"
    output_file = "Data/MNI/Motif/500/dblp.txt"


    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:

            line = line.strip()

            if line and line[0].isdigit():

                number = line.split(":")[0]

                outfile.write(f"{number}\n")

    print(f"attribute")



def read_file_and_calculate_neighbor_attributes(file_path, target_node):

    node_attributes = {}
    edges = []
    cnt=0

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if line.startswith('v'):
                node_id, attribute = int(parts[1]), int(parts[2])
                node_attributes[node_id] = attribute
            elif line.startswith('e'):
                node1, node2, _ = map(int, parts[1:])
                edges.append((node1, node2))


    neighbors = set()

    for edge in edges:
        if target_node in edge:
            neighbors.add(edge[0] if edge[1] == target_node else edge[1])
    print("attribute")

    attribute_counts = {}
    attribute_nodes = {}

    for neighbor in neighbors:
        attribute = node_attributes.get(neighbor, 0)
        if attribute not in attribute_counts:
            attribute_counts[attribute] = 0
            attribute_nodes[attribute] = []
        attribute_counts[attribute] += 1
        attribute_nodes[attribute].append(neighbor)

    for attribute, count in sorted(attribute_counts.items()):
        node_list = ", ".join(map(str, attribute_nodes[attribute]))
        print(f"attribute")


    sorted_attributes_by_count = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
    print("attribute")
    for attribute, count in sorted_attributes_by_count:
        print(f"attribute")

    return attribute_counts, attribute_nodes


def read_file_and_calculate_neighbor_attributes2(file_path, target_node, decay_factor=0.4):

    node_attributes = {}
    edges = []


    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if line.startswith('v'):
                node_id, attribute = int(parts[1]), int(parts[2])
                node_attributes[node_id] = attribute
            elif line.startswith('e'):
                node1, node2, _ = map(int, parts[1:])
                edges.append((node1, node2))

    first_order_neighbors = set()
    for edge in edges:
        if target_node in edge:
            first_order_neighbors.add(edge[0] if edge[1] == target_node else edge[1])

    second_order_neighbors = set()
    for neighbor in first_order_neighbors:
        for edge in edges:
            if neighbor in edge:
                second_order_neighbors.add(edge[0] if edge[1] == neighbor else edge[1])
    second_order_neighbors -= first_order_neighbors
    second_order_neighbors.discard(target_node)


    first_order_attribute_counts = {}
    first_order_attribute_nodes = {}
    second_order_attribute_counts = {}
    second_order_attribute_nodes = {}

    for neighbor in first_order_neighbors:
        attribute = node_attributes.get(neighbor, 0)
        if attribute not in first_order_attribute_counts:
            first_order_attribute_counts[attribute] = 0
            first_order_attribute_nodes[attribute] = []
        first_order_attribute_counts[attribute] += 1
        first_order_attribute_nodes[attribute].append(neighbor)

    for neighbor in second_order_neighbors:
        attribute = node_attributes.get(neighbor, 0)
        if attribute not in second_order_attribute_counts:
            second_order_attribute_counts[attribute] = 0
            second_order_attribute_nodes[attribute] = []
        second_order_attribute_counts[attribute] += 1
        second_order_attribute_nodes[attribute].append(neighbor)


    combined_attribute_counts = first_order_attribute_counts.copy()
    for attribute, count in second_order_attribute_counts.items():
        if attribute in combined_attribute_counts:
            combined_attribute_counts[attribute] += count * decay_factor  # 对二阶邻居的计数乘以衰减系数
        else:
            combined_attribute_counts[attribute] = count * decay_factor  # 对二阶邻居的计数乘以衰减系数


    for attribute, count in sorted(combined_attribute_counts.items()):
        node_list = ", ".join(map(str, first_order_attribute_nodes.get(attribute, []) + second_order_attribute_nodes.get(attribute, [])))
        print(f"attribute")


    print("attribute")
    sorted_combined_attributes = sorted(combined_attribute_counts.items(), key=lambda x: x[1], reverse=True)
    for attribute, count in sorted_combined_attributes:
        print(f"attribute")

    return combined_attribute_counts


def read_file_and_count_attributes(file_path):

    attribute_counts = {}


    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v'):
                parts = line.split()
                attribute = int(parts[2])
                if attribute not in attribute_counts:
                    attribute_counts[attribute] = 0
                attribute_counts[attribute] += 1


    sorted_attributes = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
    # b=torch.tensor(8)
    # a=attribute_counts.get(b.item())
    # print(a)


    for attribute, count in sorted_attributes:
        print(f"attribute")

    return sorted_attributes

from collections import defaultdict

def count_attributes(file_path):
    attribute_counts = defaultdict(int)


    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('v'):
                parts = line.split()
                if len(parts) >= 3:
                    attribute = int(parts[2])
                    attribute_counts[attribute] += 1

    for attribute, count in sorted(attribute_counts.items()):
       print("")


def result_other_model(mode,data_name, file_numb):
    path = ''
    start_time = datetime.now()
    # print()
    file_name_dict = [f"{data_name}"]
    # file_name_dict = ["dblp"]
    dataset = [{500: [0, 0, 0, 0, 0, 0]} for _ in range(file_num)]
    top_k = [500]
    fuzhu = [0.05, 0.1, 0.15000000000000002, 0.2]
    name = data_name
    res = [{f"{data_name}": dataset}]
    # res = [{"dblp": dblp}]
    # res = [{"twitch": twitch}]
    for i in range(0, file_num):
        for file_name in file_name_dict:
            for j in top_k:
                for ll in range(0, 4):
                    l = fuzhu[ll]
                    dataset[i][j][ll] = pipei_other_model(mode,file_name, i, j, l)
                print(dataset[i][j])
    storage(res)
    return res


def pipei_other_model(model,file_name,js,k,l):
    l = f"{l:.3f}"
    graph_a = read_graph_data('Data/pattern_graph/old_graph/' + file_name + '/' + str(k) + '.lg')
    graph_b = read_graph_data(
        'Data/fuzhu_result/'+model+'/'+"now1" + '/'+file_name +'/' + str(js+1 ) + '/' + str(k) + '/' + file_name + '-' + str(l) + '.lg')
    count = 0
    # for i in range(len(graph_a)):
    #     for j in range(len(graph_b)):
    #         graph_b[j][-1] = list(graph_b[j][-1])
    #         graph_b[j][-1][-1] = '0'
    #         graph_b[j][-1] = ''.join(graph_b[j][-1])
    #         if (graph_a[i] == graph_b[j]):
    #             # print(str(graph_a[i])+"   "+str(graph_b[j]))
    #             count += 1
    cn = 0
    # print(str(l) + ":")
    for i in range(len(graph_a)):
        for j in range(len(graph_b)):
            graph_b[j][-1] = list(graph_b[j][-1])
            graph_b[j][-1][-1] = '0'
            graph_b[j][-1] = ''.join(graph_b[j][-1])
            if (graph_a[i] == graph_b[j]):
                count += 1
        # if (i + 1) % 20 == 0:
        #     print(count - cn)
        #     cn = count
    return count

    # return  format(count/k, ".2f")


def averages_other_model(res,data_name,file_num):
    # file_name_dict = ["twitch"]

    file_name_dict = [data_name]
    file_ava = [{500: [0, 0, 0, 0, 0, 0]}]
    top_k = [500]
    name = data_name + '_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]


    for ff, file_name in enumerate(file_name_dict):
        for kk, k in enumerate(top_k):
            for l in range(0, 6):
                sum = 0
                for i in range(0, file_num):
                    a = res[ff][file_name][i][str(k)][l]
                    sum += float(a)
                    # res_ava[ff][f"{file_name}_ava"][0][k][l] = sum / (500)
                res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum / (500*file_num), ".2f")

    print(res_ava)
    file_name_dict = [data_name]
    file_ava = [{500: [0, 0, 0, 0, 0, 0]}]
    top_k = [500]
    name = data_name + '_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]
    # print("分开：")
    for ff, file_name in enumerate(file_name_dict):
        for kk, k in enumerate(top_k):
            for i in range(0, file_num):
                for l in range(0, 6):
                    sum = 0
                    a = res[ff][file_name][i][str(k)][l]
                    sum += float(a)
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = sum / (500)
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum / (500), ".2f")
                # print(res_ava)
    print("---------------------")



import re

def count_node_attributes(file_path):


    attribute_counts = {}


    with open(file_path, 'r') as file:
        for line in file:

            matches = re.findall(r'v \d+ (\d+)', line)
            for match in matches:

                attribute_value = int(match)
                attribute_counts[attribute_value] = attribute_counts.get(attribute_value, 0) + 1



    # print("Attribute counts:")
    sorted_attributes = sorted(attribute_counts.items(), key=lambda x: x[1], reverse=True)
    for attribute, count in sorted_attributes:
        print(f"{attribute}: {count}")
    return sorted_attributes


def count_v_blocks(file_path):
    two_v = 0
    three_v = 0
    four_v = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_block = []
    in_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue


        if line[0].isdigit():
            if in_block:

                v_count = sum(1 for entry in current_block if entry.startswith('v'))
                if v_count == 2:
                    two_v += 1
                elif v_count == 3:
                    three_v += 1
                elif v_count == 4:
                    four_v += 1
                current_block = []
            in_block = True
            current_block.append(line)
        else:
            if in_block:
                current_block.append(line)


    if in_block:
        v_count = sum(1 for entry in current_block if entry.startswith('v'))
        if v_count == 2:
            two_v += 1
        elif v_count == 3:
            three_v += 1
        elif v_count == 4:
            four_v += 1

    print(two_v)
    print(three_v)
    print(four_v)
    # return two_v, three_v, four_v



def count_attribute_mobel(file_path,label):
    count = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_block = []
    in_block = False

    for line in lines:
        line = line.strip()
        if not line:
            continue


        if line[0].isdigit() and not in_block:
            in_block = True
            current_block = [line]
        elif line[0].isdigit() and in_block:

            for entry in current_block:
                if entry.startswith('v'):
                    parts = entry.split()
                    if len(parts) >= 3 and parts[2] == label:
                        count += 1

            current_block = [line]
        else:
            if in_block:
                current_block.append(line)


    if in_block:
        for entry in current_block:
            if entry.startswith('v'):
                parts = entry.split()
                if len(parts) >= 3 and parts[2] == label:
                    count += 1

    return count



def count_attribute_file(file_path):
    with open(file_path, 'r') as file:
        labels={}
        for line in file:
            line = line.strip()
            line=line.split(' ')
            if line[0]=='v':
                labels[line[2]]=labels.get(line[2],0)+1

    print(labels)


def result_EWS1379_model(mode,data_name, file_numb,motif_num):
    path = ''
    start_time = datetime.now()
    # print()
    file_name_dict = [f"{data_name}"]
    # file_name_dict = ["dblp"]
    dataset = [{motif_num: [0, 0, 0, 0]} for _ in range(file_num)]
    top_k = [motif_num]
    fuzhu = [0.05, 0.1, 0.15000000000000002, 0.2]
    name = data_name
    res = [{f"{data_name}": dataset}]
    # res = [{"dblp": dblp}]
    # res = [{"twitch": twitch}]
    for i in range(0, file_num):
        for file_name in file_name_dict:
            for j in top_k:
                for ll in range(0, 4):
                    l = fuzhu[ll]
                    dataset[i][j][ll] = pipei_EWS1379_model(mode,file_name, i, j, l,motif_num)
                # print(dataset[i][j])
    storage(res)
    return res



def pipei_EWS1379_model(model,file_name,js,k,l,motif_num):
    l = f"{l:.3f}"
    graph_a = read_graph_data('Data/pattern_graph/old_graph/' + file_name + '/' + str(k) + '.lg')
    graph_b = read_graph_data(
        'Data/fuzhu_result/'+model+'/'+"now1" +"/"+str(k)+'/' +file_name +'/' + str(js+1 )  + '/'+'900'+'/' + file_name + '-' + str(l) + '.lg')
    count = 0
    cn = 0
    for i in range(len(graph_a)):
        for j in range(len(graph_b)):
            graph_b[j][-1] = list(graph_b[j][-1])
            graph_b[j][-1][-1] = '0'
            graph_b[j][-1] = ''.join(graph_b[j][-1])
            if (graph_a[i] == graph_b[j]):
                count += 1
    return count


def averages_EWS1379_model(res,data_name,file_num,motif_num):
    # file_name_dict = ["twitch"]

    file_name_dict = [data_name]
    file_ava = [{motif_num: [0, 0, 0, 0]}]
    top_k = [motif_num]
    name = data_name + '_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]



    for ff, file_name in enumerate(file_name_dict):
        for kk, k in enumerate(top_k):
            for l in range(0, 4):
                sum = 0
                for i in range(0, file_num):
                    a = res[ff][file_name][i][str(k)][l]
                    sum += float(a)
                    # res_ava[ff][f"{file_name}_ava"][0][k][l] = sum / (500)
                res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum / (motif_num*file_num), ".2f")

    print(res_ava[0][name][0][motif_num])
    # np.array(topk_100,res_ava[0][name][0][motif_num])
    # topk_100.
    topk_100.append(res_ava[0][name][0][motif_num])


    file_name_dict = [data_name]
    file_ava = [{motif_num: [0, 0, 0, 0]}]
    top_k = [motif_num]
    name = data_name + '_ava'
    # fuzhu = [0.3, 0.4, 0.5]#l
    # res_ava = [{"twitch_ava": twitch_ava}]
    res_ava = [{f"{name}": file_ava}]
    for ff, file_name in enumerate(file_name_dict):
        for kk, k in enumerate(top_k):
            for i in range(0, file_num):
                for l in range(0, 4):
                    sum = 0
                    a = res[ff][file_name][i][str(k)][l]
                    sum += float(a)
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = sum / (motif_num)
                    res_ava[ff][f"{file_name}_ava"][0][k][l] = format(sum / (motif_num), ".2f")
                # print(res_ava)
    # print("---------------------")


def save_to_csv(res_ava, dataset_name, motif_num, fuzhu_rates=[0.05, 0.1, 0.15, 0.2]):

    base_name = dataset_name.replace('_ava', '')
    headers = ['compression', 'CNARW', 'FF', 'RE', 'RN', 'SS']

    with open(f"{base_name}_motif{motif_num}.csv", 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)


        matrix = [
            [res_ava[algo][dataset_name][0][motif_num][rate_idx]
             for algo in headers[1:]]
            for rate_idx in range(len(fuzhu_rates))
        ]


        for rate_idx, (rate, row) in enumerate(zip(fuzhu_rates, matrix)):
            writer.writerow([f"{rate:.2f}", *map(lambda x: f"{x:.2f}", row)])



topk_100=[]
if __name__ == '__main__':
    pass

    modes = ["EWS"]
    files = ['twitch']
    file_num=5
    K_num=500
    print("EWS:")
    for file_name in files:
        data_name = file_name
        result1(data_name,file_num,K_num)
        averages1(get(),data_name,file_num,K_num)
        print()









