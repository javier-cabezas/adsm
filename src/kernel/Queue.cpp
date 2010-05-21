
#include "Queue.h"

namespace gmac {
Queue::Queue() :
    mutex(LockQueue), sem(0)
{}
}
