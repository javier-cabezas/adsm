#include "api/opencl/IOBuffer.h"

#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/Mode.h"
#include "api/opencl/hpe/Kernel.h"

#include "memory/Manager.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl { namespace hpe {

Context::Context(Mode &mode, cl_command_queue stream) :
    gmac::core::hpe::Context(mode, stream, stream, stream, stream)
{
}

Context::~Context()
{ 
    // Destroy context's private IOBuffer (if any)
    if(buffer_ != NULL) {
        TRACE(LOCAL,"Destroying I/O buffer");
    	dynamic_cast<Mode &>(mode_).destroyIOBuffer(*buffer_);
    }
}

gmacError_t Context::memset(accptr_t addr, int c, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = accelerator().memset(addr, c, size);
    trace::ExitCurrentFunction();
    return ret;
}

KernelLaunch &Context::launch(Kernel &kernel)
{
    trace::EnterCurrentFunction();
    KernelLaunch *ret = kernel.launch(dynamic_cast<Mode &>(mode_), streamLaunch_);
    ASSERTION(ret != NULL);
    trace::ExitCurrentFunction();
    return *ret;
}

}}}
