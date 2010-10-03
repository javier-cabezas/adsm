#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"

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


void Map::insert(Object &obj)
{
    lockWrite();
    ObjectMap::insert(value_type(obj.end(), &obj));
    unlock();
    trace("Adding Shared Object %p", obj.start());

    gmac::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    shared.insert(value_type(obj.end(), &obj));
    shared.unlock();
}


void Map::remove(Object &obj)
{
    ObjectMap::iterator i;
    lockWrite();
    i = ObjectMap::find(obj.end());
    if (i != end()) {
        erase(i);
    }
    unlock();

    // Shared object
    gmac::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    i = shared.find(obj.end());
    if (i != shared.end()) {
        trace("Removing Shared Object %p", obj.start());
        shared.erase(i);
    }
    shared.unlock();
    if (i != shared.end()) return;

    // Replicated object
    ObjectMap &replicated = proc.replicated();
    replicated.lockWrite();
    i = replicated.find(obj.end());
    if (i != replicated.end()) {
        trace("Removing Replicated Object %p", obj.start());
        replicated.erase(i);
    }
    replicated.unlock();
    if (i != replicated.end()) return;

    // Centralized object
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    i = centralized.find(obj.end());
    if (i != centralized.end()) {
        trace("Removing Centralized Object %p", obj.start());
        centralized.erase(i);
    }
    centralized.unlock();
}

#ifndef USE_MMAP
void Map::insertReplicated(Object &obj)
{
    trace("Adding Replicated Object %p", obj.start());
    gmac::Process &proc = parent_.process();
    ObjectMap &replicated = proc.replicated();
    replicated.lockWrite();
    replicated.insert(value_type(obj.end(), &obj));
    replicated.unlock();
    util::Logger::TRACE("Added shared object @ %p", obj.start());
}

void Map::insertCentralized(Object &obj)
{
    trace("Adding Centralized Object %p", obj.start());
    gmac::Process &proc = parent_.process();
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    centralized.insert(value_type(obj.end(), &obj));
    centralized.unlock();
    util::Logger::TRACE("Added centralized object @ %p", obj.start());
}

#endif

}}
