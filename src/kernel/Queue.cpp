
#include "Queue.h"

namespace gmac {
Queue::Queue() :
    util::Lock(LockQueue), sem(0)
{}
}
