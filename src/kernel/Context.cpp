#include "Context.h"

#include <memory/Manager.h>

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
    _releasedAll(false),
    _status(NONE)
{
    PRIVATE_SET(Context::key, this);
	_id = ++_next;
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
Context::init()
{
	TRACE("Initializing cloned context");
	Process::SharedMap::iterator i;
	Process::SharedMap &sharedMem = proc->sharedMem();
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		TRACE("Mapping Shared Region %p (%d bytes)", i->second.start(), i->second.size());
		void *devPtr;
#ifdef USE_GLOBAL_HOST
		TRACE("Using Host Translation");
		gmacError_t ret = hostMap(i->second.start(), &devPtr, i->second.size());
#else
		gmacError_t ret = malloc(&devPtr, i->second.size());
#endif
		ASSERT(ret == gmacSuccess);
		manager->remap(this, i->second.start(), devPtr, i->second.size());
		i->second.inc();
	}
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
