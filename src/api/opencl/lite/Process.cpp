#include "api/opencl/lite/Process.h"
#include "memory/Protocol.h"
#include "util/Logger.h"

#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

namespace __impl { namespace opencl { namespace lite {

Process::Process()
{
    library_t handler = USE_LIBRARY("OpenCL");
    CFATAL(handler != NULL, "Unable to get handler to OpenCL");
    addHanderl(handler);
}

Process::~Process()
{
}

gmacError_t Process::globalMalloc(memory::Object &)
{
    FATAL("Global Memory Malloc not allowed in GMAC-lite");
    return gmacErrorUnknown;
}

gmacError_t Process::globalFree(memory::Object &)
{
    FATAL("Global Memory Free not allowed in GMAC-lite");
    return gmacErrorUnknown;
}

accptr_t Process::translate(const hostptr_t addr)
{
    Mode *mode = map_.owner(addr, 0);
    if(mode == NULL) return accptr_t(0);
    memory::Object *object = mode->getObject(addr);
    if(object == NULL) return accptr_t(0);
    accptr_t ret = object->acceleratorAddr(*mode, addr);
    object->release();
    return ret;
}


memory::Protocol *Process::protocol()
{
    return NULL;
}

void Process::insertOrphan(memory::Object &)
{
    FATAL("Orphan Objects not supported in GMAC-lite");
}

core::Mode *Process::owner(const hostptr_t addr, size_t size) const
{
    return map_.owner(addr, size);
    
    return NULL;
}

Mode *Process::createMode(cl_context ctx, cl_uint numDevices, const cl_device_id *devices)
{
    Mode *ret = map_.get(ctx);
    if(ret != NULL) { ret->release(); return ret; }
    ret = new Mode(ctx, numDevices, devices);
    map_.insert(ctx, *ret);
    return ret;
}

Mode *Process::getMode(cl_context ctx)
{
    return map_.get(ctx);
}

}}}
