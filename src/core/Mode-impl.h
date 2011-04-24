#ifndef GMAC_CORE_MODE_IMPL_H_
#define GMAC_CORE_MODE_IMPL_H_

#include "config/order.h"
#include "memory/Protocol.h"
#include "trace/Tracer.h"

namespace __impl { namespace core {

inline
Mode::Mode() :
    id_(AtomicInc(Count_)),
    releasedObjects_(false)
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
memory::Protocol &Mode::protocol()
{
    return *protocol_;
}

inline
unsigned Mode::id() const
{
    return id_;
}

inline
gmacError_t Mode::error() const
{
    return error_;
}

inline
void Mode::error(gmacError_t err)
{
    error_ = err;
}

inline
bool Mode::releasedObjects() const
{
    return releasedObjects_;
}


inline void
Mode::addObject(memory::Object &obj)
{
    getObjectMap().insert(obj);
}

inline void 
Mode::removeObject(memory::Object &obj)
{
    getObjectMap().remove(obj);
}

inline memory::Object *
Mode::getObject(const hostptr_t addr, size_t size) const
{
	return getObjectMap().get(addr, size);
}

inline gmacError_t
Mode::forEachObject(memory::ObjectMap::ConstObjectOp op) const
{
    gmacError_t ret = getObjectMap().forEach(op);
	return ret;
}


} }

#endif
