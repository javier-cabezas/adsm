#include "Kernel.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl { namespace hpe {

gmacError_t
KernelLaunch::execute()
{
    trace_.init(mode_.id());
    cl_int ret = clEnqueueNDRangeKernel(stream_, f_, workDim_, globalWorkOffset_, globalWorkSize_, localWorkSize_, 0, NULL, &event_);
	clFlush(stream_);
    trace_.trace(f_, event_);

    return Accelerator::error(ret);
}

}}}
