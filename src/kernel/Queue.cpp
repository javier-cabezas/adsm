
#include "Queue.h"

namespace gmac {
Queue::Queue() :
    logger("Queue"), mutex(LockQueue), sem(0)
{}
}
