import multiprocessing as mp
import time

def worker(num):
    return num*num

def multicore():
    pool = mp.Pool()
    res = pool.map(worker, range(10))
    print(res)

if __name__ == '__main__':
    multicore()