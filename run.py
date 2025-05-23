import random
import tracemalloc

import Graphweighting
import Sampling
import time
import pandas as pd

def generate_random_integers(seed, count=10, lower=1, upper=100):
    random.seed(seed)
    return [random.randint(lower, upper) for _ in range(count)]


def sampling(mode, filename, start, end, interval,gnn_model):
    ks=[900]
    start_time = time.time()
    if mode == 'EWS':
        s_time = time.time()
        e_time = time.time()
        for K in ks:
            G = get_weightedgraph(filename, K, gnn_model)
            for js in range(1,6):
                # for k in ks:
                    name = filename + '2weight'
                    path="./Data/SamplingData/EWS/now1/"+str(K)+"/"+str(js)+"/"
                    Sampling.by_edge_weight_sampling(filename, datapath + '%s.txt' % name,
                                                      path+filename,
                                                      start,
                                                      end, interval,K,G)
    elif mode == 'RE':
        for i in range(1,6):
            Sampling.by_random_edge(filename, datapath + '%s.txt' % filename, './Data/SamplingData/RE/'+str(i)+'/%s' % filename,
                                    start,
                                    end, interval)
    elif mode == 'RNE':
        for i in range(1, 6):
            Sampling.by_random_node_edge(filename, datapath + '%s.txt' % filename, './Data/SamplingData/RNE/'+str(i)+'/%s' % filename,
                                         start, end, interval)
    elif mode == 'RN':
        for i in range(1, 6):
            Sampling.by_random_node(filename, datapath + '%s.txt' % filename, './Data/SamplingData/RN/'+str(i)+'/%s' % filename,
                                    start,
                                    end, interval)
    elif mode == 'HNE':
        for i in range(1, 6):
            Sampling.by_hybrid_node_edge(filename, datapath + '%s.txt' % filename, './Data/SamplingData/HNE/'+str(i)+'/%s' % filename,
                                         start, end, interval)
    elif mode == 'RW':
        for i in range(1, 6):
            Sampling.by_random_walk(filename, datapath + '%s.txt' % filename, './Data/SamplingData/RW/'+str(i)+'/%s' % filename,
                                    start,
                                    end, interval)
    elif mode == 'FF':
        for i in range(1, 6):
            Sampling.by_forest_fire(filename, datapath + '%s.txt' % filename, './Data/SamplingData/FF/'+str(i)+'/%s' % filename,
                                    start,
                                     end, interval)
    elif mode == 'SS':
        for i in range(1, 6):
            Sampling.by_spikyball(filename, datapath + '%s.txt' % filename, './Data/SamplingData/SS/'+str(i)+'/%s' % filename,
                                    start,
                                    end, interval)
    elif mode == 'CNARW':
        for i in range(1, 6):
            Sampling.by_common_neighbor_aware_random_walk(filename, datapath + '%s.txt' % filename, './Data/SamplingData/CNARW/'+str(i)+'/%s' % filename,
                                    start,
                                    end, interval)
    else:
        print('%s mode is not supposed!' % mode)
        quit()

    end_time = time.time()
    use_time = end_time - start_time
    df = pd.DataFrame({'sampling_time_NEW': [use_time], 'sampling_mode': [mode], 'filename': [filename], ''
                                                                                                         'time': [
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())]})
    df.to_csv('./log/exp/_sampling_time.csv', index=False, mode='a', header=False)

    if mode not in sampling_times:
        sampling_times[mode] = {}
    if filename not in sampling_times[mode]:
        sampling_times[mode][filename] = use_time
    sampling_times[mode][filename] = (sampling_times[mode][filename] + use_time) / 2

def get_weightedgraph(filename,k,gnn_model):
    return Graphweighting.get_weight_graph(datapath + '%s.txt' % filename, datapath + "%s2weight.txt" % filename,k,filename,gnn_model)


sampling_times = {}
mining_time = {}
realdata_time = {}
datapath = './Data/'
mode = ['EWS']
gnn_model='GAT'
filename = ['youtube']

def run(modes, filenames):
    print('Sampling Time test:')
    for filename in filenames:
        for mode in modes:
            print(mode + ' mode with ' + filename)
            df = pd.DataFrame({'Time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
                                       'CPU ': [filename],
                                       'Mem_perc': [mode],
                                       'Mem_use': ['0']
                                       })
            df.to_csv('./log/exp/Mem_.csv', index=False, mode='a', header=False)

            start_time = time.time()
            sampling(mode, filename, 0.050, 0.200, 0.05,gnn_model)
            end_time = time.time()
run(mode, filename)