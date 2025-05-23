# learn_mprof.py
from memory_profiler import profile
import numpy as np

@profile
def count():
    print('+++++++++++')
    a = 0;
    b = 1;
    c = a + b
    print('a+b=c')
    print('{}+{}={}'.format(a, b, c))
    print('+++++++++++')
    arr = np.ones([1000, 1000])
    print(arr.shape)
    print('+++++++++++')

if __name__ == '__main__':
    count()
