#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"
#include "DistributedObject.h"
#include "OrphanObject.h"
#include "Protocol.h"

namespace __impl { namespace memory {

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

ObjectMap::ObjectMap(const char *name) :
    gmac::util::RWLock(name)
{
}

ObjectMap::~ObjectMap()
{
}

size_t ObjectMap::size() const
{
    lockRead();
    size_t ret = Parent::size();
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

size_t ObjectMap::memorySize() const
{
    size_t total = 0;
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) total += i->second->size();
    unlock();
    return total;
}

void ObjectMap::forEach(Protocol &p, ProtocolOp op)
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) (p.*op)(*i->second);
    unlock();
}

void ObjectMap::freeObjects(Protocol &p, ProtocolOp op)
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        (p.*op)(*i->second);
        i->second->free();
    }
    unlock();
}

void ObjectMap::reallocObjects(core::Mode &mode)
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) i->second->realloc(mode);
    unlock();
}


void ObjectMap::cleanAndDestroy()
{
    lockWrite();
    for(iterator i = begin(); i != end(); i++) {
        delete i->second;
    }
    clear();
    unlock();
}


Map::Map(const char *name, core::Mode &parent) :
    ObjectMap(name), parent_(parent)
{
}

Map::~Map()
{
    TRACE(LOCAL,"Cleaning Memory Map");
    clean();
}

Map &
Map::operator =(const Map &)
{
    FATAL("Assigment of memory maps is not supported");
    return *this;
}

const Object *Map::getObjectRead(const void *addr) const
{
    const core::Process &proc = parent_.process();
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
    const core::Process &proc = parent_.process();
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
    core::Process &proc = parent_.process();
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
    TRACE(LOCAL,"Adding Shared Object %p", obj.addr());

    core::Process &proc = parent_.process();
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
        TRACE(LOCAL,"Removing Local Object %p", obj.addr());
        erase(i);

        ASSERTION(ObjectMap::find(obj.end()) == end());
    }
    unlock();

    // Shared object
    core::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
    shared.lockWrite();
    i = shared.find(obj.end());
	bool done = (i != shared.end());
    if(done) {
        TRACE(LOCAL,"Removing Shared Object %p", obj.addr());
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
        TRACE(LOCAL,"Removing Replicated Object %p", obj.addr());
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
        TRACE(LOCAL,"Removing Centralized Object %p", obj.addr());
        centralized.erase(i);
    }
    centralized.unlock();
	if(done) return;

    // Orphan object
    ObjectMap &orphans = proc.orphans();
    orphans.lockWrite();
    i = orphans.find(obj.end());
    if (i != orphans.end()) {
        TRACE(LOCAL,"Removing Orphan Object %p", obj.addr());
        orphans.erase(i);
    }
    orphans.unlock();

}

#ifndef USE_MMAP
void Map::insertReplicated(Object &obj)
{
    TRACE(LOCAL,"Adding Replicated Object %p", obj.addr());
    core::Process &proc = parent_.process();
    ObjectMap &replicated = proc.replicated();
    replicated.lockWrite();
    replicated.insert(value_type(obj.end(), &obj));
    replicated.unlock();
    TRACE(LOCAL, "Added shared object @ %p", obj.addr());
}

void Map::insertCentralized(Object &obj)
{
    TRACE(LOCAL,"Adding Centralized Object %p", obj.addr());
    core::Process &proc = parent_.process();
    ObjectMap &centralized = proc.centralized();
    centralized.lockWrite();
    centralized.insert(value_type(obj.end(), &obj));
    centralized.unlock();
    TRACE(LOCAL, "Added centralized object @ %p", obj.addr());
}

void Map::addOwner(core::Process &proc, core::Mode &mode)
{
    ObjectMap &replicated = proc.replicated();
    iterator i;
    replicated.lockWrite();
    for(i = replicated.begin(); i != replicated.end(); i++) {
        i->second->lockWrite();
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->addOwner(mode);
        i->second->unlock();
    }
    replicated.unlock();
}


void Map::removeOwner(core::Process &proc, core::Mode &mode)
{
    ObjectMap &replicated = proc.replicated();
    iterator i;
    replicated.lockWrite();
    for(i = replicated.begin(); i != replicated.end(); i++) {
        i->second->lockWrite();
        memory::DistributedObject *obj =
                dynamic_cast<memory::DistributedObject *>(i->second);
        obj->removeOwner(mode);
        i->second->unlock();
    }
    replicated.unlock();
}
#endif

void Map::makeOrphans()
{
    TRACE(LOCAL,"Converting remaining objects to orphans");
    ObjectMap &orphans = parent_.process().orphans();
    orphans.lockWrite();
    lockWrite();
    for(iterator i = begin(); i != end(); i++) {
        OrphanObject *orphan = new __impl::memory::OrphanObject(*i->second);
        orphans.insert(value_type(orphan->end(), orphan));
        delete i->second;
    }
    unlock();
    orphans.unlock();

    clean();
}


}}
