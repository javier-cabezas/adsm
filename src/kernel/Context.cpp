#include "Context.h"

#include <memory/Manager.h>
#include <config/paraver.h>

extern gmac::memory::Manager *manager;

namespace gmac {


void
Context::Init()
{
	PRIVATE_INIT(Context::key, NULL);
	PRIVATE_SET(Context::key, NULL);
}


PRIVATE(Context::key);

unsigned Context::_next = 0;

Context::Context(Accelerator &acc) :
    _acc(acc),
    _kernels(),
    _releasedRegions(),
    _releasedAll(false)
{
    PRIVATE_SET(Context::key, this);
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

Context *
Context::current()
{
    Context * ctx;
    ctx = static_cast<Context *>(PRIVATE_GET(Context::key));
    if (ctx == NULL) ctx = proc->create();
    return ctx;
}

void
Context::initThread()
{
    PRIVATE_SET(key, NULL);
}

void
Context::destroy()
{
    // Set the current context before each Context destruction (since it is sequential)
    PRIVATE_SET(key, this);
    _acc.destroy(this);
    PRIVATE_SET(key, NULL);
}

}
