
#include "Queue.h"

namespace gmac {
Queue::Queue() :
    util::Lock(LockQueue), sem(0)
{}

void Queue::push(Mode *mode)
{
    lock();
    _queue.push_back(mode);
    unlock();
    sem.post();
}

Mode * Queue::pop()
{
    sem.wait();
    lock();
    assertion(_queue.empty() == false);
    Mode *ret = _queue.front();
    _queue.pop_front();
    unlock();
    return ret;
}

ThreadQueue::ThreadQueue() :
    util::Lock(LockThreadQueue)
{
    queue = new Queue();
}

ThreadQueue::~ThreadQueue()
{
    delete queue;
}


}
