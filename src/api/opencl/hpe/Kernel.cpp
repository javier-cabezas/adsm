#include "Kernel.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl { namespace hpe {


KernelConfig &KernelConfig::operator=(const KernelConfig &config)
{
    if(this == &config) return *this;

    workDim_ = config.workDim_;
    globalWorkOffset_ = NULL;
    globalWorkSize_ = NULL;
    localWorkSize_ = NULL;

    if(config.globalWorkOffset_) globalWorkOffset_ = new size_t[workDim_];
    if(config.globalWorkSize_) globalWorkSize_ = new size_t[workDim_];
    if(config.localWorkSize_) localWorkSize_ = new size_t[workDim_];

    for(unsigned i = 0; i < workDim_; i++) {
        if(globalWorkOffset_) globalWorkOffset_[i] = config.globalWorkOffset_[i];
        if(globalWorkSize_) globalWorkSize_[i] = config.globalWorkSize_[i];
        if(localWorkSize_) localWorkSize_[i] = config.localWorkSize_[i];
    }

    resize(config.size());

    for(unsigned i = 0; i < config.size(); i++) {
        setArgument(config[i].ptr(), config[i].size(), i);
    }

    return *this;
}


gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    for(unsigned i = 0; i < size(); i++) {
        TRACE(LOCAL, "Setting param %u @ %p ("FMT_SIZE")", i, at(i).ptr(), at(i).size());
        cl_int ret = clSetKernelArg(f_, i, at(i).size(), at(i).ptr());
        CFATAL(ret == CL_SUCCESS, "OpenCL Error setting parameters: %d", ret);
    }


    trace_.init(mode_.id());
    cl_event event;
    cl_int ret = clEnqueueNDRangeKernel(stream_, f_, workDim_, globalWorkOffset_, globalWorkSize_, localWorkSize_, 0, NULL, &event);
	clFlush(stream_);
    trace_.trace(f_, event);

    return Accelerator::error(ret);
}

}}}
