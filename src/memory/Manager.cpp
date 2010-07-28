#include "Manager.h"
#include "manager/BatchManager.h"
#include "manager/LazyManager.h"
#include "manager/RollingManager.h"

#include <strings.h>


namespace gmac { namespace memory {

static int Manager::__count = 0;
static Manager *Manager::__manager = NULL;

Manager::Manager()
{
    trace("Memory manager starts");
    assertion(__count == 0);
}

Manager::~Manager()
{
    trace("Memory manager finishes");
    assertion(__count == 0);
}


gmacError_t
Manager::alloc(void ** addr, size_t size)
{
    gmacError_t ret;
    // Create new shared object
    SharedObject *object = new SharedObject(protocol, size);
    *addr = object->addr();
    if(*addr == NULL) return gmacErrorMemoryAllocation;

    // Insert object into memory maps
    Map *local = Map::current();
    local->lockWrite();
    local->insert(object);
    local->unlock();

    return gmacSuccess;
}

gmacError_t
Manager::free(void * addr)
{
    gmacError_t ret = gmacSuccess;
    Map *local = Map::current();
    local->lockWrite();
    Object *object = local->find(addr);
    if(object != NULL)  {
        local->remove(object);
        delete object;
    }
    else ret = gmacErrorInvalid;
    local->unlock();
    return ret;
}

gmacError_t
Manager::adquire()
{
    return gmacSuccess;
}

gmacError_t
Manager::release()
{
    return gmacSuccess;
}

void
Manager::read(void *addr)
{
}

void
Manager::write(void *addr)
{
}

}}
