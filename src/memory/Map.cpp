#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"
#include "Protocol.h"

namespace __impl { namespace memory {

const Object *ObjectMap::mapFind(const void *addr, size_t size) const
{
    ObjectMap::const_iterator i;
    const Object *ret = NULL;
    lockRead();
    const uint8_t *limit = (const uint8_t *)addr + size;
    i = upper_bound(addr);
    if(i != end() && i->second->addr() <= limit) ret = i->second;
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

const Object *ObjectMap::get(const void *addr, size_t size) const
{
    const Object *ret = NULL;
    ret = mapFind(addr, size);
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

void ObjectMap::forEach(const core::Mode &mode, ModeOp op) const
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

const Object *Map::get(const ObjectMap &map, const uint8_t *&base, const void *addr, size_t size) const
{
    const Object *ret = map.mapFind(addr, size);
    if(ret == NULL) return ret;
    if(base == NULL || ret->addr() < base) {
        base = ret->addr();
        return ret;
    }
    return NULL;
}

const Object *Map::get(const void *addr, size_t size) const
{    
    const Object *ret = NULL;
    const uint8_t *base = NULL;
    // Lookup in the current map
    ret = get(*this, base, addr, size);
    if(base == addr) goto exit_func;

    // Check global maps
    const core::Process &proc = parent_.process();
    const Object *obj = NULL;
    obj = get(proc.shared(), base, addr, size);
    if(obj != NULL) ret = obj;
    if(base == addr) goto exit_func;
    obj = get(proc.global(), base, addr, size);
    if(obj != NULL) ret = obj;
    if(base == addr) goto exit_func;
    obj = get(proc.orphans(), base, addr, size);
    if(obj != NULL) ret = obj;

exit_func:
    if(ret != NULL) ret->use();
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
	void *addr = obj.addr();

	bool ret = ObjectMap::remove(obj);
	
    // Shared object
    core::Process &proc = parent_.process();
    ObjectMap &shared = proc.shared();
	ret = shared.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Shared Object %p", addr);
		return true;
	}

    // Replicated object
    ObjectMap &global = proc.global();
	ret = global.remove(obj);
	if(ret == true) {
		TRACE(LOCAL,"Removed Global Object %p", addr);
		return true;
	}

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
    orphans.insert(obj);
}

void Map::addOwner(core::Process &proc, core::Mode &mode)
{
	ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); i++) {
        i->second->addOwner(mode);
    }
    global.unlock();
}


void Map::removeOwner(core::Process &proc, const core::Mode &mode)
{
    ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); i++) {        
        i->second->removeOwner(mode);
    }
    global.unlock();
}

}}
