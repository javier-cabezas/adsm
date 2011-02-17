#include "core/Process.h"
#include "core/Mode.h"
#include "core/Context.h"

#include "Map.h"
#include "Object.h"
#include "Protocol.h"

namespace __impl { namespace memory {

Object *ObjectMap::mapFind(const hostptr_t addr, size_t size) const
{
    ObjectMap::const_iterator i;
    Object *ret = NULL;
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

bool ObjectMap::remove(Object &obj)
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

Object *ObjectMap::get(const hostptr_t addr, size_t size) const
{
    Object *ret = NULL;
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

gmacError_t ObjectMap::forEach(ObjectOp op) const
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*op)();
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}

gmacError_t ObjectMap::forEach(ConstObjectOp op) const
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*op)();
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}

gmacError_t ObjectMap::forEach(core::Mode &mode, ModeOp op) const
{
    const_iterator i;
    lockRead();
    for(i = begin(); i != end(); i++) {
        gmacError_t ret = (i->second->*op)(mode);
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return gmacSuccess;
}

Map::Map(const char *name, core::Mode &parent) :
    ObjectMap(name), parent_(parent)
{
}

Map::~Map()
{
    TRACE(LOCAL,"Cleaning Memory Map");
    //TODO: actually clean the memory map
}

Object *Map::get(const ObjectMap &map, hostptr_t &base, const hostptr_t addr, size_t size) const
{
    Object *ret = map.mapFind(addr, size);
    if(ret == NULL) return ret;
    if(base == NULL || ret->addr() < base) {
        base = ret->addr();
        return ret;
    }
    return NULL;
}

Object *Map::get(const hostptr_t addr, size_t size) const
{
    Object *ret = NULL;
    hostptr_t base = NULL;
    // Lookup in the current map
    ret = get(*this, base, addr, size);

    // Check global maps
    const core::Process &proc = parent_.process();
    Object *obj = NULL;

    if(base == addr) goto exit_func;

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


bool Map::remove(Object &obj)
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


void Map::removeOwner(core::Process &proc, core::Mode &mode)
{
    ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); i++) {
        i->second->removeOwner(mode);
    }
    global.unlock();

    ObjectMap &shared = proc.shared();
    iterator j;
    shared.lockWrite();
    for(j = shared.begin(); j != shared.end(); j++) {        
        j->second->removeOwner(mode);
    }
    shared.unlock();
}

}}
