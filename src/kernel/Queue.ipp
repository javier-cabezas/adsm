#ifndef __KERNEL_QUEUE_IPP_
#define __KERNEL_QUEUE_IPP_

inline void
Queue::push(Mode *mode)
{
    mutex.lock();
    _queue.push_back(mode);
    mutex.unlock();
    sem.post();
}

inline Mode *
Queue::pop()
{
    sem.wait();
    mutex.lock();
    assertion(_queue.empty() == false);
    Mode *ret = _queue.front();
    _queue.pop_front();
    mutex.unlock();
    return ret;
}

#endif
