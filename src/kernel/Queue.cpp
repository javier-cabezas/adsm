
#include "Queue.h"

namespace gmac {
Queue::Queue() :
    mutex(paraver::queueLock), sem(0)
{}
}
