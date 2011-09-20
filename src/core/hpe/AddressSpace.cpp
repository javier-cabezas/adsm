#include "Process.h"
#include "Mode.h"
#include "Context.h"

#include "memory/ObjectMap.h"
#include "memory/Object.h"

namespace __impl { namespace core { namespace hpe {

AddressSpace::AddressSpace(const char *name, Process &parent) :
    Parent(name, parent)
{
}

AddressSpace::~AddressSpace()
{
    TRACE(LOCAL,"Cleaning Memory AddressSpace");
    //TODO: actually clean the memory map
}

Process &
AddressSpace::getProcess()
{
    return static_cast<Process &>(parent_);
}

const Process &
AddressSpace::getProcess() const
{
    return static_cast<Process &>(parent_);
}

memory::Object *
AddressSpace::getObject(const hostptr_t addr, size_t size) const
{
    memory::Object *ret = NULL;
    // Lookup in the current map
    ret = mapFind(addr, size);

    // Exit if we have found it
    if(ret != NULL) goto exit_func;

    // Check global maps
    {
        ret = getProcess().shared().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = getProcess().global().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
        ret = getProcess().orphans().mapFind(addr, size);
        if (ret != NULL) goto exit_func;
    }

exit_func:
    if(ret != NULL) ret->incRef();
    return ret;
}

bool AddressSpace::addObject(memory::Object &obj)
{
    TRACE(LOCAL,"Adding Shared Object %p", obj.addr());
    bool ret = Parent::addObject(obj);
    if(ret == false) return ret;
    memory::ObjectMap &shared = getProcess().shared();
    ret = shared.addObject(obj);
    return ret;
}


bool AddressSpace::removeObject(memory::Object &obj)
{
    bool ret = Parent::removeObject(obj);
    hostptr_t addr = obj.addr();
    // Shared object
    memory::ObjectMap &shared = getProcess().shared();
    ret = shared.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Shared Object %p", addr);
        return true;
    }

    // Replicated object
    memory::ObjectMap &global = getProcess().global();
    ret = global.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Global Object %p", addr);
        return true;
    }

    // Orphan object
    memory::ObjectMap &orphans = getProcess().orphans();
    ret = orphans.removeObject(obj);
    if(ret == true) {
        TRACE(LOCAL,"Removed Orphan Object %p", addr);
    }

    return ret;
}

void AddressSpace::addOwner(Process &proc, Mode &mode)
{
    memory::ObjectMap &global = proc.global();
    iterator i;
    global.lockWrite();
    for(i = global.begin(); i != global.end(); i++) {
        i->second->addOwner(mode);
    }
    global.unlock();
}

void AddressSpace::removeOwner(Process &proc, Mode &mode)
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
