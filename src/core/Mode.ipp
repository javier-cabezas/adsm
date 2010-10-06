#ifndef GMAC_CORE_MODE_IPP
#define GMAC_CORE_MODE_IPP

#include "memory/Map.h"

namespace gmac {

inline void ContextMap::add(THREAD_ID id, Context *ctx)
{
    lockWrite();
    Parent::insert(Parent::value_type(id, ctx));
    unlock();
}

inline Context *ContextMap::find(THREAD_ID id)
{
    lockRead();
    Parent::iterator i = Parent::find(id);
    Context *ret = NULL;
    if(i != end()) ret = i->second;
    unlock();
    return ret;
}

inline void ContextMap::remove(THREAD_ID id)
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
    gmacError_t ret;
    lockRead();
    for(i = begin(); i != end(); i++) {
        ret = i->second->sync();
        if(ret == gmacSuccess) continue;
        unlock(); return ret;
    }
    unlock();
    return ret;
}

inline void Mode::init()
{
    gmac::util::Private<Mode>::init(key);
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

inline void Mode::inc()
{
    count_++;
}

inline void Mode::destroy()
{
    count_--;
    if(count_ == 0) delete this;
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

#ifndef USE_MMAP
inline void
Mode::addReplicatedObject(memory::Object &obj)
{
    map_.insertReplicated(obj);
}

inline void
Mode::addCentralizedObject(memory::Object &obj)
{
    map_.insertCentralized(obj);
}

#endif

inline const memory::Object *
Mode::getObjectRead(const void *addr) const
{
    const memory::Object *obj = map_.getObjectRead(addr);
    return obj;
}

inline memory::Object *
Mode::getObjectWrite(const void *addr)
{
    memory::Object *obj = map_.getObjectWrite(addr);
    return obj;
}

inline void
Mode::putObject(const memory::Object &obj)
{
    map_.putObject(obj);
}

inline const memory::Map &
Mode::objects()
{
    return map_;
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

}

#endif
