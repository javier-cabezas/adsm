#include "Context.h"

#include <memory/Manager.h>

#include <threads.h>

extern gmac::memory::Manager *manager;

namespace gmac {

void contextInit()
{
	PRIVATE_INIT(gmac::Context::key, NULL);
	PRIVATE_SET(gmac::Context::key, NULL);
}


PRIVATE(Context::key);

//std::list<Context *> *Context::list = NULL;

Context::Context(Accelerator &acc) : acc(acc)
{
	PRIVATE_SET(key, NULL);
}

void Context::init()
{
	TRACE("Initializing cloned context");
	//memory::Map::SharedList::const_iterator i;
	Process::SharedMap::iterator i;
	Process::SharedMap &sharedMem = proc->sharedMem();
	//for(i = memory::Map::shared().begin(); i != memory::Map::shared().end(); i++) {
	for(i = sharedMem.begin(); i != sharedMem.end(); i++) {
		TRACE("Mapping Shared Region %p (%d bytes)", i->second.start(), i->second.size());
		void *devPtr;
#ifdef USE_GLOBAL_HOST
		TRACE("Using Host Translation");
		gmacError_t ret = hostMap(i->second.start(), &devPtr, i->second.size());
#else
		gmacError_t ret = malloc(&devPtr, i->second.size());
#endif
		assert(ret == gmacSuccess);
		manager->remap(this, i->second.start(), devPtr, i->second.size());
		i->second.inc();
	}
}
}
