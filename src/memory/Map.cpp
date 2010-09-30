#include "Map.h"
#include "Object.h"

#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

namespace gmac { namespace memory {

Object * ObjectMap::mapFind(const void *addr) const
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    lockRead();
    i = upper_bound(addr);
    if(i != end() && i->second->start() <= addr)
        ret = i->second;
    unlock();
    return ret;
}


const Object *ObjectMap::getObjectRead(const void *addr) const
{
    const Object *ret = NULL;
    ret = mapFind(addr);
    if (ret != NULL) ret->lockRead();
    return ret;
}

Object *ObjectMap::getObjectWrite(const void *addr) const
{
    Object *ret = NULL;
    ret = mapFind(addr);
    if (ret != NULL) ret->lockWrite();
    return ret;
}

void ObjectMap::putObject(const Object &obj) const
{
    obj.unlock();
}


Map::~Map()
{
    trace("Cleaning Memory Map");
    clean();
}


const Object *Map::getObjectRead(const void *addr) const
{
    const gmac::Process &proc = parent_.process();
    Object *ret = NULL;
    ret = mapFind(addr);
    if(ret == NULL)  ret = proc.shared().mapFind(addr);
#ifndef USE_MMAP
    if(ret == NULL)  ret = proc.replicated().mapFind(addr);
    if(ret == NULL)  ret = proc.centralized().mapFind(addr);
#endif
    if (ret != NULL) ret->lockRead();
    return ret;
}

Object *Map::getObjectWrite(const void *addr) const
{
    const gmac::Process &proc = parent_.process();
    Object *ret = NULL;
    ret = mapFind(addr);
    if(ret == NULL)  ret = proc.shared().mapFind(addr);
#ifndef USE_MMAP
    if(ret == NULL)  ret = proc.replicated().mapFind(addr);
    if(ret == NULL)  ret = proc.centralized().mapFind(addr);
#endif
    if (ret != NULL) ret->lockWrite();
    return ret;
}

void Map::clean()
{
    gmac::Process &proc = parent_.process();
	ObjectMap::iterator i;
    ObjectMap &global = proc.shared();
	lockWrite();
	for(i = begin(); i != end(); i++) {
		trace("Cleaning Object %p", i->second->start());
		if(&global != this) global.lockWrite();
		global.erase(i->first);
		if(&global != this) global.unlock();
	}
	clear();
	unlock();
}


void Map::insert(Object *obj)
{
    lockWrite();
    ObjectMap::insert(value_type(obj->end(), obj));
    unlock();

    gmac::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    shared.insert(value_type(obj->end(), obj));
    shared.unlock();
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

#if 0
    ObjectMap &__shared = proc->shared();
    __shared.lockWrite();
    i = __shared.find(obj->end());
    if(i != __shared.end()) __shared.erase(i);
    __shared.unlock();
#endif

    gmac::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    i = shared.find(obj->end());
    if(i != shared.end()) shared.erase(i);
    shared.unlock();

}

#ifndef USE_MMAP
void Map::insertReplicated(Object* obj)
{
    gmac::Process &proc = parent_.process();
    ObjectMap &replicated = proc.shared();
    replicated.lockWrite();
    replicated.insert(value_type(obj->end(), obj));
    replicated.unlock();
    util::Logger::TRACE("Added shared object @ %p", obj->start());
}

Object *Map::removeReplicated(const void *addr)
{
    gmac::Process &proc = parent_.process();
    ObjectMap::iterator i;
    ObjectMap &replicated = proc.shared();
    replicated.lockWrite();
    i = replicated.upper_bound(addr);
    Object *obj = i->second;
    assertion(i != replicated.end() && obj->start() == addr);
    replicated.erase(i);
    replicated.unlock();
    util::Logger::TRACE("Removed shared object @ %p", obj->start());
    return obj;
}

void Map::insertCentralized(Object* obj)
{
    gmac::Process &proc = parent_.process();
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    centralized.insert(value_type(obj->end(), obj));
    centralized.unlock();
    util::Logger::TRACE("Added centralized object @ %p", obj->start());
}


void Map::removeCentralized(Object *obj)
{
    gmac::Process &proc = parent_.process();
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    ObjectMap::iterator i = centralized.find(obj->end());
    assertion(i != centralized.end());
    centralized.erase(i);
    centralized.unlock();
    util::Logger::TRACE("Removed centralized region @ %p", obj->start());
}
#endif

}}
