#include "trace/Tracer.h"
#include "util/Logger.h"

#include "Accelerator.h"

namespace __impl { namespace core { namespace hpe {

MapAllocations::MapAllocations() :
    gmac::util::RWLock("AcceleratorMapAlloc")
{
}

void
MapAllocations::insert(hostptr_t key, accptr_t val, size_t size)
{
    lockWrite();
    ASSERTION(MapAlloc::find(key) == end());
    MapAlloc::insert(MapAlloc::value_type(key, PairAlloc(val, size)));
    unlock();
}

void
MapAllocations::erase(hostptr_t key, size_t size)
{
    lockWrite();
    MapAlloc::iterator it = MapAlloc::find(key);
    ASSERTION(it != end());
    MapAlloc::erase(it);
    unlock();
}

bool
MapAllocations::find(hostptr_t key, accptr_t &val, size_t &size)
{
    lockRead();
    MapAlloc::const_iterator it = MapAlloc::find(key);
    bool ret = false;
    if (it != MapAlloc::end()) {
        val  = it->second.first;
        size = it->second.second;
        ret = true;
    }
    unlock();
    return ret;
}

Accelerator::Accelerator(int n) :
    id_(n), load_(0)
{
}

Accelerator::~Accelerator()
{
}

void Accelerator::registerMode(Mode &mode)
{
    TRACE(LOCAL,"Registering Execution Mode %p to Accelerator", &mode);
    trace::EnterCurrentFunction();
    load_++;
    trace::ExitCurrentFunction();
}

void Accelerator::unregisterMode(Mode &mode)
{
    TRACE(LOCAL,"Unregistering Execution Mode %p", &mode);
    trace::EnterCurrentFunction();
    load_--;
    trace::ExitCurrentFunction();
}

}}}
