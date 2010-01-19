
#include "Queue.h"

namespace gmac {
namespace kernel {
Queue::Queue() :
    mutex(paraver::queueLock), sem(0)
{}
}
}
