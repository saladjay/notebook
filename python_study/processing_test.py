import multiprocessing as mp
import threading as td
import queue

def job(q):
    print(type(q))
    res = 0
    for i in range(10000000):
        res += i
    q.put(res)

if __name__ == '__main__&mp.Queue':
    q1 = mp.Queue()
    q2 = mp.Queue()
    q1.put(1)
    p1 = mp.Process(target=job, args=(q1,)) # 只能使用mp.Queue, 不能使用queue.Queue
    p2 = mp.Process(target=job, args=(q2,)) # 只有一个参数时，应该加上逗号
    p1.start()
    p2.start()
    p1.join()
    p2.join()


import multiprocessing as mp
import threading as td
import time

def job(q):
    res = 0
    for i in range(1000000):
        res += i+i**2+i**3
    q.put(res) # queue

def multicore():
    q = mp.Queue()
    p1 = mp.Process(target=job, args=(q,))
    p2 = mp.Process(target=job, args=(q,))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print('multicore:' , res1+res2)

def normal():
    res = 0
    for _ in range(2):
        for i in range(1000000):
            res += i+i**2+i**3
    print('normal:', res)

def multithread():
    q = mp.Queue()
    t1 = td.Thread(target=job, args=(q,))
    t2 = td.Thread(target=job, args=(q,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    res1 = q.get()
    res2 = q.get()
    print('multithread:', res1+res2)

if __name__ == '__main__':
    from viztracer import VizTracer
    tracer = VizTracer()
    tracer.start()
    st = time.time()
    normal()
    st1= time.time()
    print('normal time:', st1 - st)
    multithread()
    st2 = time.time()
    print('multithread time:', st2 - st1)
    multicore()
    print('multicore time:', time.time()-st2)

    # Something happens here
    tracer.stop()
    tracer.save('./viztracer_output.json') # also takes output_file as an optional argument