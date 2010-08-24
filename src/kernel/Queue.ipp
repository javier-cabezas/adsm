#ifndef __KERNEL_QUEUE_IPP_
#define __KERNEL_QUEUE_IPP_

inline void
Queue::push(Mode *mode)
{
    lock();
    _queue.push_back(mode);
    unlock();
    sem.post();
}

inline Mode *
Queue::pop()
{
    sem.wait();
    lock();
    assertion(_queue.empty() == false);
    Mode *ret = _queue.front();
    _queue.pop_front();
    unlock();
    return ret;
}

#endif
