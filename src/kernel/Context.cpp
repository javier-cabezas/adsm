#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

extern gmac::memory::Manager *manager;

namespace gmac {

Context::Context(Mode *mode) :
    util::RWLock(LockContext),
    kernels()
{
    addThreadTid(0x10000000 + mode->id());
    pushState(Idle, 0x10000000 + mode->id());
}

Context::~Context()
{
    KernelMap::iterator it;
    for (it = kernels.begin(); it != kernels.end(); it++) {
        delete it->second;
    }
}

}
