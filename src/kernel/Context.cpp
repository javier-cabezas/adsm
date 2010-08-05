#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

extern gmac::memory::Manager *manager;

namespace gmac {

unsigned Context::_next = 0;

Context::Context(Accelerator *acc) :
    util::Reference(),
    util::RWLock(LockContext),
    _error(gmacSuccess),
    _kernels(),
    _releasedObjects(),
    _releasedAll(false)
{
	_id = ++_next;
    addThreadTid(0x10000000 + _id);
    pushState(Idle, 0x10000000 + _id);
}

Context::~Context()
{
    KernelMap::iterator it;

    for (it = _kernels.begin(); it != _kernels.end(); it++) {
        delete it->second;
    }
}


void
Context::clearKernels()
{
    KernelMap::iterator i;
    for(i = _kernels.begin(); i != _kernels.end(); i++) {
        delete i->second;
    }
    _kernels.clear();
}

}
