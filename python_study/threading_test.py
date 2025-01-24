import threading
from queue import Queue
import copy
import time
from viztracer import VizTracer
def job(l, q):
    res = sum(l)
    q.put(res)

def multithreading(l):
    q = Queue()
    threads = []
    for i in range(4):
        t = threading.Thread(target=job, args=(l[i], q), name='T%i' % i)
        t.start()
        threads.append(t)

    [t.join() for t in threads]
    total = 0
    for _ in range(4):
        total += q.get()
    print(total)

def normal(l):
    total = sum(l)
    print(total)

if __name__ == '__main__':
    tracer = VizTracer()
    tracer.start()
    l = list(range(1000000))
    
    ls1 = l*4
    s_t = time.time()
    normal(ls1)
    print('normal: ',time.time()-s_t)
    ls = [l,l,l,l]
    s_t = time.time()
    multithreading(ls)
    print('multithreading: ', time.time()-s_t)

    # Something happens here
    tracer.stop()
    tracer.save('./viztracer_output.json') # also takes output_file as an optional argument