#include "Map.h"
#include "Object.h"

#include <kernel/Process.h>
#include <kernel/Mode.h>
#include <kernel/Context.h>

#include <gmac/init.h>

namespace gmac { namespace memory {

Map::~Map()
{
    trace("Cleaning Memory Map");
    clean();
}

Object * Map::mapFind(ObjectMap &map, const void *addr)
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    map.lockRead();
    i = map.upper_bound(addr);
    if(i != map.end() && i->second->start() <= addr)
        ret = i->second;
    map.unlock();
    return ret;
}

Object *Map::globalFind(const void *addr)
{
    return mapFind(proc->global(), addr);
}

#ifndef USE_MMAP
Object *Map::sharedFind(const void *addr)
{
    return mapFind(proc->shared(), addr);
}
#endif

Object *Map::find(const void *addr)
{
    Object *ret = NULL;
    ret = localFind(addr);
    if(ret == NULL) { ret = globalFind(addr); }
#ifndef USE_MMAP
    if(ret == NULL) { ret = sharedFind(addr); }
#endif
    return ret;
}



void Map::clean()
{
	ObjectMap::iterator i;
    ObjectMap &__global = proc->global();
	lockWrite();
	for(i = begin(); i != end(); i++) {
		trace("Cleaning Object %p", i->second->start());
		if(&__global != this) __global.lockWrite();
		__global.erase(i->first);
		if(&__global != this) __global.unlock();
	}
	clear();
	unlock();
}


void Map::insert(Object *obj)
{
    lockWrite();
    ObjectMap::insert(value_type(obj->end(), obj));
    unlock();

    ObjectMap &__global = proc->global();
    __global.lockWrite();
    __global.insert(value_type(obj->end(), obj));
    __global.unlock();
}


void Map::remove(Object *obj)
{
    ObjectMap::iterator i;
    trace("Removing Object %p", obj->start());
    lockWrite();
    i = ObjectMap::find(obj->end());
    if(i != end()) {
        erase(i);
    }
    unlock();

    ObjectMap &__shared = proc->shared();
    __shared.lockWrite();
    i = __shared.find(obj->end());
    if(i != __shared.end()) __shared.erase(i);
    __shared.unlock();


    ObjectMap &__global = proc->global();
    __global.lockWrite();
    i = __global.find(obj->end());
    if(i != __global.end()) __global.erase(i);
    __global.unlock();

}

#ifndef USE_MMAP
void Map::insertShared(Object* obj)
{
    ObjectMap &__shared = proc->shared();
    __shared.lockWrite();
    __shared.insert(value_type(obj->end(), obj));
    __shared.unlock();
    util::Logger::TRACE("Added shared object @ %p", obj->start());
}

Object *Map::removeShared(const void *addr)
{
    ObjectMap::iterator i;
    ObjectMap &__shared = proc->shared();
    __shared.lockWrite();
    i = __shared.upper_bound(addr);
    Object *obj = i->second;
    assertion(i != __shared.end() && obj->start() == addr);
    __shared.erase(i);
    __shared.unlock();
    util::Logger::TRACE("Removed shared object @ %p", obj->start());
    return obj;
}

void Map::insertGlobal(Object* obj)
{
    ObjectMap &__global = proc->global();
    __global.lockWrite();
    __global.insert(value_type(obj->end(), obj));
    __global.unlock();
    util::Logger::TRACE("Added global object @ %p", obj->start());
}


void Map::removeGlobal(Object *obj)
{
    ObjectMap &__global = proc->global();
    __global.lockWrite();
    ObjectMap::iterator i = __global.find(obj->end());
    assertion(i != __global.end());
    __global.erase(i);
    __global.unlock();
    util::Logger::TRACE("Removed global region @ %p", obj->start());
}
#endif

}}
