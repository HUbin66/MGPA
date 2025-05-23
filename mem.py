
# !/usr/bin/python
import pandas as pd
import psutil
import os, datetime, time

record_interval = 0.5  # unit is second


def getMemCpu():
    data = psutil.virtual_memory()
    total = data.total
    free = data.available
    memory = str(int(round(data.percent)))
    memory_use = str(int(round((total - free) / 1024 / 1024)))
    cpu = str(psutil.cpu_percent(interval=record_interval))
    return (cpu, memory,memory_use)

interval = 5
while True:
    info = getMemCpu()
    print(info)
    df = pd.DataFrame({'Time': [time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())],
                       'CPU ': [info[0]],
                       'Mem_perc': [info[1]],
                       'Mem_use': [info[2]]
                       })
    df.to_csv('./log/exp/Mem_.csv', index=False, mode='a', header=False)
    time.sleep(interval)