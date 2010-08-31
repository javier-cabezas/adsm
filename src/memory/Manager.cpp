#include <memory/Manager.h>
#include <memory/Map.h>
#include <memory/Object.h>

#include <memory/protocol/Lazy.h>

#include <strings.h>

namespace gmac { namespace memory {

int Manager::__count = 0;
Manager *Manager::__manager = NULL;

Manager *Manager::create()
{
    __count++;
    if(__manager != NULL) return __manager;
    gmac::util::Logger::TRACE("Creating new Memory Manager");
    __manager = new Manager();
    return __manager;
}

void Manager::destroy()
{
    __count--;
    if(__count > 0) return;
    gmac::util::Logger::TRACE("Destroying Memory Manager");
    delete __manager;
    __manager = NULL;
}

Manager *Manager::get()
{
    gmac::util::Logger::ASSERTION(__manager != NULL);
    return __manager;
}

Manager::Manager()
{
    trace("Memory manager starts");
    assertion(__count == 1);
    
    // Create protocol
    if(strcasecmp(paramProtocol, "Rolling") == 0) {
        fatal("Protocol not supported yet");
    }
    else if(strcasecmp(paramProtocol, "Lazy") == 0) {
        protocol = new protocol::Lazy();
    }
    else {
        fatal("Memory Coherence Protocol not defined");
    }
}


gmacError_t
Manager::alloc(void ** addr, size_t size)
{
    gmacError_t ret;
    // Create new shared object
    Object *object = protocol->createObject(size);
    *addr = object->addr();
    if(*addr == NULL) {
        delete object;
        return gmacErrorMemoryAllocation;
    }

    // Insert object into memory maps
    Mode::current()->addObject(object);

    return gmacSuccess;
}

gmacError_t
Manager::free(void * addr)
{
    gmacError_t ret = gmacSuccess;
    Object *object = Mode::current()->findObject(addr);
    if(object != NULL)  {
        Mode::current()->removeObject(object);
        delete object;
    }
    else ret = gmacErrorInvalidValue;
    return ret;
}

gmacError_t Manager::acquire()
{
    const Map &map = Mode::current()->objects();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++)
        protocol->acquire(*i->second);
    return gmacSuccess;
}

gmacError_t Manager::release()
{
    const Map &map = Mode::current()->objects();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++)
        protocol->release(*i->second);
    return gmacSuccess;
}

gmacError_t Manager::invalidate()
{
    return gmacSuccess;
}

gmacError_t Manager::adquire(void *addr, size_t size)
{
    return gmacSuccess;
}

gmacError_t Manager::release(void *addr, size_t size)
{
    return gmacSuccess;
}

gmacError_t Manager::invalidate(void *addr, size_t size)
{
    return gmacSuccess;
}

bool Manager::read(void *addr)
{
    Object *obj = gmac::Mode::current()->findObject(addr);
    if(obj == NULL) return false;
    trace("Read access for object %p", obj->addr());
    if(protocol->read(*obj, addr) != gmacSuccess) return false;
    return true;
}

bool Manager::write(void *addr)
{
    Object *obj = gmac::Mode::current()->findObject(addr);
    if(obj == NULL) return false;
    trace("Write access for object %p", obj->addr());
    if(protocol->write(*obj, addr) != gmacSuccess) return false;
    return true;
}

}}
