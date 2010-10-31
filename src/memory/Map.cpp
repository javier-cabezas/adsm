#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"
#include "OrphanObject.h"

namespace gmac { namespace memory {

Object * ObjectMap::mapFind(const void *addr) const
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
    lockRead();
    i = upper_bound(addr);
    if(i != end() && i->second->addr() <= addr)
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
    if(ret == NULL)  ret = proc.orphans().mapFind(addr);
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
    if(ret == NULL)  ret = proc.orphans().mapFind(addr);
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
    trace("Adding Shared Object %p", obj.addr());

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
        trace("Removing Local Object %p", obj.addr());
        erase(i);

        assertion(ObjectMap::find(obj.end()) == end());
    }
    unlock();

    // Shared object
    gmac::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    i = shared.find(obj.end());
	bool done = (i != shared.end());
    if(done) {
        trace("Removing Shared Object %p", obj.addr());
        shared.erase(i);
    }
    shared.unlock();
    if(done) return;

    // Replicated object
    ObjectMap &replicated = proc.replicated();
    replicated.lockWrite();
    i = replicated.find(obj.end());
	done = (i != replicated.end());
    if(done) {
        trace("Removing Replicated Object %p", obj.addr());
        replicated.erase(i);
    }
    replicated.unlock();
    if(done) return;

    // Centralized object
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    i = centralized.find(obj.end());
	done = (i != centralized.end());
    if(done) {
        trace("Removing Centralized Object %p", obj.addr());
        centralized.erase(i);
    }
    centralized.unlock();
	if(done) return;

    // Orphan object
    ObjectMap &orphans = proc.orphans();
    orphans.lockWrite();
    i = orphans.find(obj.end());
    if (i != orphans.end()) {
        trace("Removing Orphan Object %p", obj.addr());
        orphans.erase(i);
    }
    orphans.unlock();

}

#ifndef USE_MMAP
void Map::insertReplicated(Object &obj)
{
    trace("Adding Replicated Object %p", obj.addr());
    gmac::Process &proc = parent_.process();
    ObjectMap &replicated = proc.replicated();
    replicated.lockWrite();
    replicated.insert(value_type(obj.end(), &obj));
    replicated.unlock();
    util::Logger::TRACE("Added shared object @ %p", obj.addr());
}

void Map::insertCentralized(Object &obj)
{
    trace("Adding Centralized Object %p", obj.addr());
    gmac::Process &proc = parent_.process();
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    centralized.insert(value_type(obj.end(), &obj));
    centralized.unlock();
    util::Logger::TRACE("Added centralized object @ %p", obj.addr());
}
#endif

void Map::makeOrphans()
{
    trace("Converting remaining objects to orphans");
    ObjectMap &orphans = parent_.process().orphans();
    orphans.lockWrite();
    lockWrite();
    for(iterator i = begin(); i != end(); i++) {
        OrphanObject *orphan = new OrphanObject(*i->second);
        orphans.insert(value_type(orphan->end(), orphan));
        delete i->second;
    }
    unlock();
    orphans.unlock();

    clean();
}

}}
