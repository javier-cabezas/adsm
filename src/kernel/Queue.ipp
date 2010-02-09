#ifndef __KERNEL_QUEUE_IPP_
#define __KERNEL_QUEUE_IPP_

inline void
Queue::push(Context *ctx)
{
    mutex.lock();
    _queue.push_back(ctx);
    mutex.unlock();
    sem.post();
}

inline Context *
Queue::pop()
{
    sem.wait();
    mutex.lock();
    assert(_queue.empty() == false);
    Context *ret = _queue.front();
    _queue.pop_front();
    mutex.unlock();
    return ret;
}

#endif
