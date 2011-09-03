#include "Process.h"
#include "Mode.h"
#include "Context.h"

#include "memory/ObjectMap.h"
#include "memory/Object.h"

namespace __impl { namespace core { namespace hpe {

Map::Map(const char *name, Mode &parent) :
    memory::ObjectMap(name), parent_(parent)
{
}

Map::~Map()
{
    TRACE(LOCAL,"Cleaning Memory Map");
    //TODO: actually clean the memory map
}

memory::Object *
Map::get(const hostptr_t addr, size_t size) const
{
    memory::Object *ret = NULL;
    // Lookup in the current map
    ret = mapFind(addr, size);

    // Exit if we have found it
    if(ret != NULL) goto exit_func;

    // Check global maps
    {
        const Process &proc = parent_.getProcess();

        ret = proc.shared().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = proc.global().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = proc.global().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
    }

exit_func:
    if(ret != NULL) ret->incRef();
    return ret;
}

bool Map::insert(memory::Object &obj)
{
    TRACE(LOCAL,"Adding Shared Object %p", obj.addr());
    bool ret = memory::ObjectMap::insert(obj);
    if(ret == false) return ret;
    memory::ObjectMap &shared = parent_.getProcess().shared();
    ret = shared.insert(obj);
    return ret;
}


bool Map::remove(memory::Object &obj)
{
    bool ret = Parent::remove(obj);
    hostptr_t addr = obj.addr();
    // Shared object
    Process &proc = parent_.getProcess();
    memory::ObjectMap &shared = proc.shared();
    ret = shared.remove(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Shared Object %p", addr);
        return true;
    }

    // Replicated object
    memory::ObjectMap &global = proc.global();
    ret = global.remove(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Global Object %p", addr);
        return true;
    }

    // Orphan object
    memory::ObjectMap &orphans = proc.orphans();
    ret = orphans.remove(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Orphan Object %p", addr);
    }

    return ret;
}

void Map::addOwner(Process &proc, Mode &mode)
{
    memory::ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); i++) {
        i->second->addOwner(mode);
    }
    global.unlock();
}

void Map::removeOwner(Process &proc, Mode &mode)
{
    memory::ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for (i = global.begin(); i != global.end(); i++) {
        i->second->removeOwner(mode);
    }
    global.unlock();

    /*
    memory::ObjectMap &shared = proc.shared();
    iterator j;
    shared.lockWrite();
    for (j = shared.begin(); j != shared.end(); j++) {
        // Owners already removed in Mode::cleanUp
        j->second->removeOwner(mode);
    }
    shared.unlock();
    */
}

}}}
