#include "Manager.h"
#include "manager/BatchManager.h"
#include "manager/LazyManager.h"
#include "manager/RollingManager.h"

#include <strings.h>


namespace gmac { namespace memory {

static int Manager::__count = 0;
static Manager *Manager::__manager = NULL;

static Manager *Manager::create()
{
    __count++;
    if(__manager != NULL) return __manager;
    __manager = new Manager();
    return __manager;
}

static void Manager::destroy()
{
    __count--;
    if(__count > 0) return;
    delete __manager;
    __manager = NULL;
}

static Manager *Manager::get()
{
    assertion(__manager != NULL);
    return __manager;
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
