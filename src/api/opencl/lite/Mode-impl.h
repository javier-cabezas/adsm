#ifndef GMAC_API_OPENCL_LITE_MODE_IMPL_H_
#define GMAC_API_OPENCL_LITE_MODE_IMPL_H_

#include "api/opencl/IOBuffer.h"
#include "api/opencl/lite/Process.h"

namespace __impl { namespace opencl { namespace lite {

inline
QueueSet::QueueSet() :
    gmac::util::RWLock("QueueMap")
{}

inline
QueueSet::~QueueSet() { }

inline
void QueueSet::insert(cl_command_queue queue)
{
    lockWrite();
    Parent::insert(queue); 
    unlock();
}

inline
bool QueueSet::exists(cl_command_queue queue)
{
    bool ret = false;
    lockRead();
    if(Parent::find(queue) != Parent::end()) ret = true;
    unlock();
    return ret;
}

inline
void QueueSet::remove(cl_command_queue queue)
{
    lockWrite();
    Parent::erase(queue);
    unlock();
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

inline
gmacError_t Mode::hostAlloc(hostptr_t &, size_t)
{
    FATAL("Host Memory allocation not supported in GMAC/Lite");
    return gmacErrorUnknown;
}

inline
gmacError_t Mode::hostFree(hostptr_t)
{
    FATAL("Host Memory release not supported in GMAC/Lite");
    return gmacErrorUnknown;
}


inline
accptr_t Mode::hostMapAddr(const hostptr_t)
{
    FATAL("Host Memory translation is not supported in GMAC/Lite");
    return accptr_t(0);
}


inline
core::IOBuffer &Mode::createIOBuffer(size_t size)
{
    core::IOBuffer *ret;
    void *addr = ::malloc(size);
    ret = new IOBuffer(*this, hostptr_t(addr), size, false);
    return *ret;
}

inline
void Mode::destroyIOBuffer(core::IOBuffer &buffer)
{
    ::free(buffer.addr());
    delete &buffer;
}

inline
gmacError_t Mode::waitForEvent(cl_event event)
{
    //TODO: implement waiting for events
    return gmacErrorUnknown;
}

inline
gmacError_t Mode::eventTime(uint64_t &t, cl_event start, cl_event end)
{
    gmacError_t ret = gmacSuccess;
    return ret; 
}

inline
gmacError_t Mode::releaseObjects()
{
    releasedObjects_ = true;
    return error_;
}

inline
gmacError_t Mode::acquireObjects()
{
    cl_int ret = clFinish(active_);
    releasedObjects_ = false;
    return error(ret);
}

inline
void Mode::insertOrphan(memory::Object &obj)
{
    FATAL("Orphan objects not supported in GMAC/Lite");
}

}}}

#endif
