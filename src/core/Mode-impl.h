#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

#include "config/order.h"
#include "memory/Object.h"
#include "memory/ObjectMap.h"
#include "memory/Protocol.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

inline
Mode::Mode() :
    util::Reference("Mode"),
    gmac::util::SpinLock("Mode"),
    id_(AtomicInc(Count_)),
    modifiedObjects_(false),
    releasedObjects_(false)
#ifdef USE_VM
    , bitmap_(*this)
#endif
{
    TRACE(LOCAL,"Creating Execution Mode %p", this);
    trace::StartThread(THREAD_T(id_), "GPU");
    SetThreadState(THREAD_T(id_), trace::Idle);
    protocol_ = memory::ProtocolInit(0);
}

inline
Mode::~Mode()
{
    delete protocol_;
    trace::EndThread(THREAD_T(id_));
    TRACE(LOCAL,"Destroying Execution Mode %p", this);
}

inline
memory::Protocol &Mode::getProtocol()
{
    return *protocol_;
}

inline
unsigned Mode::id() const
{
    return id_;
}

inline
bool Mode::hasModifiedObjects() const
{
    lock();
    bool ret = modifiedObjects_;
    unlock();
    return ret;
}

inline
void Mode::invalidateObjects()
{
    lock();
    modifiedObjects_ = false;
    unlock();
}


inline
void Mode::modifiedObjects()
{
    lock();
    modifiedObjects_ = true;
    releasedObjects_ = false;
    unlock();
}

inline
bool Mode::releasedObjects() const
{
    lock();
    bool ret = releasedObjects_;
    unlock();
    return ret;
}

inline void
Mode::addObject(memory::Object &obj)
{
    getAddressSpace().insert(obj);
    modifiedObjects();
}

inline void
Mode::removeObject(memory::Object &obj)
{
    getAddressSpace().remove(obj);
}

inline memory::Object *
Mode::getObject(const hostptr_t addr, size_t size) const
{
    return getAddressSpace().get(addr, size);
}

inline gmacError_t
Mode::forEachObject(gmacError_t (memory::Object::*f)(void))
{
    gmacError_t ret = getAddressSpace().forEachObject(f);
    return ret;
}

template <typename T>
inline gmacError_t
Mode::forEachObject(gmacError_t (memory::Object::*f)(T &), T &param)
{
    gmacError_t ret = getAddressSpace().forEachObject(f, param);
    return ret;
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


} }

#endif
