#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

extern gmac::memory::Manager *manager;

namespace gmac {


void
Context::Init()
{
    gmac::util::Private::init(Context::key);
    Context::key.set(NULL);
}


gmac::util::Private Context::key;

unsigned Context::_next = 0;

Context::Context(Accelerator *acc) :
    util::Reference(),
    util::RWLock(LockContext),
    _error(gmacSuccess),
    _acc(acc),
    _kernels(),
    _releasedRegions(),
    _releasedAll(false)
{
    _mm = new  memory::Map();
    Context::key.set(this);
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

    if(_mm != NULL) delete _mm;
}

Context *
Context::current()
{
    Context * ctx;
    ctx = static_cast<Context *>(Context::key.get());
    if (ctx == NULL) ctx = proc->create();
    return ctx;
}

void
Context::initThread()
{
    key.set(NULL);
}

void
Context::cleanup()
{
    // Delete this memory map first 
    delete _mm;
    _mm = NULL;
    // Set the current context before each Context destruction (since it is sequential)
    key.set(this);
    _acc->destroy(this);
    proc->remove(this);
    key.set(NULL);
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
