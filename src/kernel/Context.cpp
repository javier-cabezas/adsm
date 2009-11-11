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

std::list<Context *> *Context::list = NULL;

Context::Context(Accelerator &acc) : acc(acc)
{
	PRIVATE_SET(key, NULL);
	if(list == NULL) list = new std::list<Context *>();
	list->push_back(this);
}

void Context::init()
{
	TRACE("Initializing cloned context");
	memory::Map::SharedList::const_iterator i;
	for(i = memory::Map::shared().begin(); i != memory::Map::shared().end(); i++) {
		TRACE("Mapping Shared Region %p (%d bytes)", (*i)->start(), (*i)->size());
		void *devPtr;
#ifdef USE_GLOBAL_HOST
		TRACE("Using Host Translation");
		gmacError_t ret = host_map((*i)->start(), &devPtr, (*i)->size());
#else
		gmacError_t ret = malloc(&devPtr, (*i)->size());
#endif
		assert(ret == gmacSuccess);
		manager->remap(this, (*i)->start(), devPtr, (*i)->size());
	}
}
}
