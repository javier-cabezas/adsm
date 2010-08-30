#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

namespace gmac {

Context::Context(Accelerator *acc) :
    util::RWLock(LockContext),
    acc(acc)
{
    addThreadTid(0x10000000 + mode->id());
    pushState(Idle, 0x10000000 + mode->id());
}

Context::~Context() { }

}
