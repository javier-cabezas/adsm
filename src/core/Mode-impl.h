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

inline gmacError_t ContextMap::sync()
{
    Parent::iterator i;
    gmacError_t ret = gmacSuccess;
    lockRead();
    for(i = begin(); i != end(); i++) {
        ret = i->second->sync();
        if(ret != gmacSuccess) {
            unlock();
            return ret;
        }
    }
    unlock();
    return ret;
}

inline void Mode::cleanUp() const
{
    map_.forEach(*this, &memory::Object::removeOwner);
    memory::Map::removeOwner(Process::getInstance(), *this);
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

inline unsigned Mode::accId() const
{
    return acc_->id();
}

inline bool Mode::integrated() const
{
    return acc_->integrated();
}

inline void
Mode::addObject(memory::Object &obj)
{
    map_.insert(obj);
}

inline void 
Mode::removeObject(const memory::Object &obj)
{
    map_.remove(obj);
}

inline const memory::Object *
Mode::getObject(const void *addr, size_t size) const
{
	return map_.get(addr, size);
}

inline void
Mode::forEachObject(memory::ObjectMap::ObjectOp op) const
{
	return map_.forEach(op);
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
inline memory::vm::Bitmap &
Mode::dirtyBitmap()
{
    return *_bitmap;
}

inline const memory::vm::Bitmap &
Mode::dirtyBitmap() const
{
    return *_bitmap;
}
#endif

inline bool
Mode::releasedObjects() const
{
    return releasedObjects_;
}

inline void
Mode::releaseObjects()
{
    releasedObjects_ = true;
}

inline void
Mode::acquireObjects()
{
    releasedObjects_ = false;
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
