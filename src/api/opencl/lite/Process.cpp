#include "api/opencl/lite/Process.h"
#include "memory/Protocol.h"
#include "util/Logger.h"

namespace __impl { namespace opencl { namespace lite {

Process::Process()
{}

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
    //TODO: implement
    return accptr_t(0);
}


memory::Protocol &Process::protocol()
{
    FATAL("Global Memory Protocol not implemented in GMAC-lite");
    return *(memory::Protocol *)0;
}

void Process::insertOrphan(memory::Object &)
{
    FATAL("Orphan Objects not supported in GMAC-lite");
}

core::Mode *Process::owner(const hostptr_t addr, size_t size) const
{
    // TODO: Implement
    return NULL;
}

Mode *Process::createMode(cl_context ctx, cl_uint numDevices, const cl_device_id *devices)
{
    Mode *ret = map_.get(ctx);
    if(ret != NULL) return ret;
    ret = new Mode(ctx, numDevices, devices);
    map_.insert(ctx, *ret);
    return ret;
}

}}}
