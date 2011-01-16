#ifndef GMAC_CORE_MODE_IPP
#define GMAC_CORE_MODE_IPP

#include "memory/Map.h"
#include "memory/Object.h"

#include "core/Process.h"

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


inline gmacError_t Mode::cleanUp()
{
    gmacError_t ret = map_.forEach(*this, &memory::Object::removeOwner);
    memory::Map::removeOwner(Process::getInstance(), *this);
#ifdef USE_SUBBLOCK_TRACKING
    hostBitmap_.cleanUp();
#else
#ifdef USE_VM
    acceleratorBitmap_.cleanUp();
#endif
#endif
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
Mode::forEachObject(memory::ObjectMap::ConstObjectOp op) const
{
    gmacError_t ret = map_.forEach(op);
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

#ifdef USE_SUBBLOCK_TRACKING
inline memory::vm::BitmapHost &
Mode::hostDirtyBitmap()
{
    return hostBitmap_;
}

inline const memory::vm::BitmapHost &
Mode::hostDirtyBitmap() const
{
    return hostBitmap_;
}

#else

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
    switchOut();
    return error_;
}

inline gmacError_t 
Mode::acquireObjects()
{
    switchIn();
    releasedObjects_ = false;
    error_ = contextMap_.waitForCall();
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

}}

#endif
