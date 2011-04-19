#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

#ifdef DEBUG
#include <fstream>
#include <sstream>
#endif

#include "memory/Object.h"

#include "Context.h"

namespace __impl { namespace core {

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

inline gmacError_t ContextMap::waitForCall(core::KernelLaunch &launch)
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

inline void Mode::cleanUpContexts()
{
    contextMap_.clean();
}

inline void Mode::init()
{
    util::Private<Mode>::init(key);
}

inline void Mode::initThread()
{
    key.set(NULL);
}

inline bool
Mode::hasCurrent()
{
    return key.get() != NULL;
}

inline memory::Protocol &Mode::protocol()
{
    return *protocol_;
}

inline unsigned Mode::id() const
{
    return id_;
}

inline
Accelerator &
Mode::getAccelerator() const
{
    return *acc_;
}

inline void
Mode::addObject(memory::Object &obj)
{
    map_.insert(obj);
}

inline void 
Mode::removeObject(memory::Object &obj)
{
    map_.remove(obj);
}

inline memory::Object *
Mode::getObject(const hostptr_t addr, size_t size) const
{
	return map_.get(addr, size);
}

inline gmacError_t
Mode::forEachObject(gmacError_t (memory::Object::*f)(void) const) const
{
    gmacError_t ret = map_.forEachObject(f);
    return ret;
}

inline gmacError_t
Mode::forEachObject(gmacError_t (memory::Object::*f)(void))
{
    gmacError_t ret = map_.forEachObject(f);
    return ret;
}


inline gmacError_t
Mode::error() const
{
    return error_;
}

inline void
Mode::error(gmacError_t err)
{
    error_ = err;
}

#ifdef USE_VM
inline memory::vm::Bitmap&
Mode::getBitmap()
{
    return bitmap_;
}

inline const memory::vm::Bitmap&
Mode::getBitmap() const
{
    return bitmap_;
}
#endif

inline bool
Mode::releasedObjects() const
{
    return releasedObjects_;
}

inline gmacError_t 
Mode::releaseObjects()
{
    switchIn();
    releasedObjects_ = true;
    error_ = contextMap_.prepareForCall();
#ifdef USE_VM
    bitmap_.release();
#endif
    switchOut();
    return error_;
}

inline gmacError_t
Mode::dump(std::string _name, uint8_t *addr)
{
#ifdef DEBUG
    static int i = 0;
    std::stringstream name;
    name << _name << "-" << i++;

    if (addr == NULL) {
        map_.dumpObjects(name.str());
    } else {
#if 0
        std::ofstream out(name.str().c_str(), std::ios_base::trunc);
        ASSERTION(out.good());
        memory::Object *obj = getObject(addr);
        ASSERTION(obj != NULL);
        obj->dump(out);
        out.close();
#endif
    }
#endif
    return gmacSuccess;
}

inline gmacError_t 
Mode::acquireObjects()
{
    switchIn();
    releasedObjects_ = false;
    switchOut();
    return error_;
}

inline Process &
Mode::process()
{
    return proc_;
}

inline const Process &
Mode::process() const
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

}}

#endif
