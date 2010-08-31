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
        object->lock();
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
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        object.lock();
        protocol->acquire(object);
        object.unlock();
    }
    return gmacSuccess;
}

gmacError_t Manager::release()
{
    const Map &map = Mode::current()->objects();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        object.lock();
        protocol->release(object);
        object.unlock();
    }
    return gmacSuccess;
}

gmacError_t Manager::invalidate()
{
    const Map &map = Mode::current()->objects();
    Map::const_iterator i;
    for(i = map.begin(); i != map.end(); i++) {
        Object &object = *i->second;
        object.lock();
        protocol->invalidate(object);
        object.unlock();
    }
    return gmacSuccess;
}

gmacError_t Manager::adquire(void *addr, size_t size)
{
    uint8_t *ptr = (uint8_t *)addr;
    do {
        Object *obj = Mode::current()->findObject(ptr);
        obj->lock();
        protocol->invalidate(*obj);
        ptr += obj->size();
        obj->unlock();
    } while(ptr < (uint8_t *)addr + size);
    return gmacSuccess;
}

gmacError_t Manager::release(void *addr, size_t size)
{
    uint8_t *ptr = (uint8_t *)addr;
    do {
        Object *obj = Mode::current()->findObject(ptr);
        obj->lock();
        protocol->release(*obj);
        ptr += obj->size();
        obj->unlock();
    } while(ptr < (uint8_t *)addr + size);
    return gmacSuccess;
}

gmacError_t Manager::invalidate(void *addr, size_t size)
{
    uint8_t *ptr = (uint8_t *)addr;
    do {
        Object *obj = Mode::current()->findObject(ptr);
        obj->lock();
        protocol->acquire(*obj);
        ptr += obj->size();
        obj->unlock();
    } while(ptr < (uint8_t *)addr + size);
    return gmacSuccess;
}

bool Manager::read(void *addr)
{
    bool ret = true;
    Object *obj = gmac::Mode::current()->findObject(addr);
    if(obj == NULL) return false;
    trace("Read access for object %p", obj->addr());
    obj->lock();
    if(protocol->read(*obj, addr) != gmacSuccess) ret = false;
    obj->unlock();
    return ret;
}

bool Manager::write(void *addr)
{
    bool ret = true;
    Object *obj = gmac::Mode::current()->findObject(addr);
    if(obj == NULL) return false;
    trace("Write access for object %p", obj->addr());
    obj->lock();
    if(protocol->write(*obj, addr) != gmacSuccess) ret = false;
    obj->unlock();
    return ret;
}

}}
