#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"
#include "OwnerMap.h"
#include "Protocol.h"

namespace __impl { namespace memory {

const Object *ObjectMap::mapFind(const void *addr) const
{
    ObjectMap::const_iterator i;
    const Object *ret = NULL;
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

bool ObjectMap::insert(Object &obj)
{
	lockWrite();
	std::pair<iterator, bool> ret = Parent::insert(value_type(obj.end(), &obj));
	if(ret.second == true) obj.use();
	unlock();
	return ret.second;
}

bool ObjectMap::remove(const Object &obj)
{
	lockWrite();
	iterator i = find(obj.end());
	bool ret = (i != end());
	if(ret == true) {
		obj.release();
		Parent::erase(i);
	}
	unlock();
	return ret;
}

const Object *ObjectMap::get(const void *addr) const
{
    const Object *ret = NULL;
    ret = mapFind(addr);
	if(ret != NULL) ret->use();
    return ret;
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

void ObjectMap::forEach(ObjectOp op) const
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) (i->second->*op)();
    unlock();
}

void ObjectMap::forEach(core::Mode &mode, ModeOp op) const
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) (i->second->*op)(mode);
    unlock();
}

#if 0
void ObjectMap::reallocObjects(core::Mode &mode)
{
    iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) i->second->realloc(mode);
    unlock();
}
#endif


Map::Map(const char *name, core::Mode &parent) :
    ObjectMap(name), parent_(parent)
{
}

Map::~Map()
{
    TRACE(LOCAL,"Cleaning Memory Map");
	//TODO: actually clean the memory map
}

Map &
Map::operator =(const Map &)
{
    FATAL("Assigment of memory maps is not supported");
    return *this;
}

const Object *Map::get(const void *addr) const
{
    const core::Process &proc = parent_.process();
    const Object *ret = NULL;
    ret = mapFind(addr);
    if(ret == NULL)  ret = proc.shared().mapFind(addr);
    if(ret == NULL)  ret = proc.replicated().mapFind(addr);
    //if(ret == NULL)  ret = proc.centralized().mapFind(addr);
    if(ret == NULL)  ret = proc.orphans().mapFind(addr);
	if (ret != NULL) ret->use();
    return ret;
}

bool Map::insert(Object &obj)
{    
    TRACE(LOCAL,"Adding Shared Object %p", obj.addr());
	bool ret = ObjectMap::insert(obj);
	if(ret == false) return ret;
    core::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
	ret = shared.insert(obj);
    return ret;
}


bool Map::remove(const Object &obj)
{
#if defined(DEBUG)
	void *addr = obj.addr();
#endif

	bool ret = ObjectMap::remove(obj);
	if(ret == false) WARNING("Map did not remove local object %p", addr);
	
    // Shared object
    core::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
	ret = shared.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Shared Object %p", addr);
		return true;
	}

    // Replicated object
    ObjectMap &replicated = proc.replicated();
	ret = replicated.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Replicated Object %p", addr);
		return true;
	}

#if 0
    // Centralized object
    ObjectMap &centralized = proc.centralized();
	ret = centralized.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Centralized Object %p", addr);
		return true;
	}
#endif

    // Orphan object
    ObjectMap &orphans = proc.orphans();
	ret = orphans.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Orphan Object %p", addr);
	}
    
	return ret;
}

void Map::insertOrphan(Object &obj)
{
    core::Process &proc = core::Process::getInstance();
    ObjectMap &orphans = proc.orphans();
    orphans.lockWrite();
    orphans.insert(obj);
    orphans.unlock();
}

void Map::addOwner(core::Process &proc, core::Mode &mode)
{
	ObjectMap &replicated = proc.replicated();
    iterator i;
    replicated.lockWrite();
    for(i = replicated.begin(); i != replicated.end(); i++) {
        i->second->addOwner(mode);
    }
    replicated.unlock();
}


void Map::removeOwner(core::Process &proc, core::Mode &mode)
{
    ObjectMap &replicated = proc.replicated();
    iterator i;
    replicated.lockWrite();
    for(i = replicated.begin(); i != replicated.end(); i++) {        
        i->second->removeOwner(mode);
    }
    replicated.unlock();
}

}}
