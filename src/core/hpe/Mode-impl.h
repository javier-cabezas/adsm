#ifndef GMAC_CORE_HPE_MODE_IMPL_H_
#define GMAC_CORE_HPE_MODE_IMPL_H_

#include "memory/Object.h"

#include "core/hpe/Accelerator.h"
#include "core/hpe/Process.h"
#include "core/hpe/Context.h"

namespace __impl { namespace core { namespace hpe {

inline void ContextMap::add(THREAD_T id, Context *ctx)
{
    lockWrite();
    Parent::insert(Parent::value_type(id, ctx));
    unlock();
}

inline Context *ContextMap::find(THREAD_T id)
{
    lockRead();
    Parent::iterator i = Parent::find(id);
    Context *ret = NULL;
    if(i != end()) ret = i->second;
    unlock();
    return ret;
}

inline void ContextMap::remove(THREAD_T id)
{
    lockWrite();
    Parent::erase(id);
    unlock();
}

inline void ContextMap::clean()
{
    Parent::iterator i;
    lockWrite();
    for(i = begin(); i != end(); i++) delete i->second;
    Parent::clear();
    unlock();
}

inline gmacError_t ContextMap::prepareForCall()
{
    Parent::iterator i;
    gmacError_t ret = gmacSuccess;
    lockRead();
    for(i = begin(); i != end(); i++) {
        ret = i->second->prepareForCall();
        if(ret != gmacSuccess) break;
    }
    unlock();
    return ret;
}

inline gmacError_t ContextMap::waitForCall()
{
    Parent::iterator i;
    gmacError_t ret = gmacSuccess;
    lockRead();
    for(i = begin(); i != end(); i++) {
        ret = i->second->waitForCall();
        if(ret != gmacSuccess) break;
    }
    unlock();
    return ret;
}

inline gmacError_t ContextMap::waitForCall(KernelLaunch &launch)
{
    Parent::iterator i;
    gmacError_t ret = gmacSuccess;
    lockRead();
    for(i = begin(); i != end(); i++) {
        ret = i->second->waitForCall(launch);
        if(ret != gmacSuccess) break;
    }
    unlock();
    return ret;
}

inline
memory::ObjectMap &Mode::getObjectMap()
{
    return map_;
}

inline
const memory::ObjectMap &Mode::getObjectMap() const
{
    return map_;
}


inline void Mode::cleanUpContexts()
{
    contextMap_.clean();
}

inline void Mode::init()
{
    util::Private<Mode>::init(key);
    util::Private<Process>::init(parent);
}

inline void Mode::initThread(Process &proc)
{
    key.set(NULL);
    parent.set(&proc);
}

inline bool
Mode::hasCurrent()
{
    return key.get() != NULL;
}

inline
void Mode::insertOrphan(memory::Object &obj)
{
    proc_.insertOrphan(obj);
}

inline
Accelerator &
Mode::getAccelerator() const
{
    return *acc_;
}

#ifdef USE_VM
inline memory::vm::BitmapShared &
Mode::acceleratorDirtyBitmap()
{
    return acceleratorBitmap_;
}

inline const memory::vm::BitmapShared &
Mode::acceleratorDirtyBitmap() const
{
    return acceleratorBitmap_;
}
#endif


inline Process &Mode::process()
{
    return proc_;
}

inline const Process &Mode::process() const
{
    return proc_;
}

inline void
Mode::memInfo(size_t &free, size_t &total)
{
    switchIn();
    acc_->memInfo(free, total);
    switchOut();
}

}}}

#endif
