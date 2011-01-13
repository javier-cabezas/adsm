#include "Kernel.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl {

Kernel::Kernel(const core::KernelDescriptor & k, cl_kernel kernel) :
    core::Kernel(k), f_(kernel)
{
}

core::KernelLaunch *
Kernel::launch(core::KernelConfig & _c)
{
    KernelConfig & c = static_cast<KernelConfig &>(_c);

    KernelLaunch * l = new opencl::KernelLaunch(*this, c);
    return l;
}

KernelConfig::KernelConfig() :
    globalWorkOffset_(NULL),
    globalWorkSize_(NULL),
    localWorkSize_(NULL)
{
}

KernelConfig::KernelConfig(cl_uint workDim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize, cl_command_queue stream) :
    core::KernelConfig(),
    workDim_(workDim),
    globalWorkOffset_(NULL),
    globalWorkSize_(NULL),
    localWorkSize_(NULL),
    stream_(stream)
{
    if (globalWorkOffset) globalWorkOffset_ = new size_t[workDim];
    if (globalWorkSize) globalWorkSize_ = new size_t[workDim];
    if (localWorkSize) localWorkSize_ = new size_t[workDim];

    for (unsigned i = 0; i < workDim; i++) {
        if (globalWorkOffset) globalWorkOffset_[i] = globalWorkOffset[i];
        if (globalWorkSize) globalWorkSize_[i] = globalWorkSize[i];
        if (localWorkSize) localWorkSize_[i] = localWorkSize[i];
    }
}

KernelConfig::KernelConfig(const KernelConfig &config)
{
    ASSERTION(0);
}

KernelConfig::~KernelConfig()
{
    if (globalWorkOffset_) delete [] globalWorkOffset_;
    if (globalWorkSize_) delete [] globalWorkSize_;
    if (localWorkSize_) delete [] localWorkSize_;
}

KernelLaunch::KernelLaunch(const Kernel & k, const KernelConfig & c) :
    core::KernelLaunch(),
    opencl::KernelConfig(c),
    f_(k.f_)
{
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    unsigned i = 0;
    for (std::vector<core::Argument>::const_iterator it = begin(); it != end(); it++) {
        TRACE(LOCAL, "Setting param %d @ %p ("FMT_SIZE")", i, it->ptr(), it->size());
        cl_int ret = clSetKernelArg(f_, i, it->size(), it->ptr());
        CFATAL(ret == CL_SUCCESS, "OpenCL Error setting parameters: %d", ret);
        i++;
    }

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = textures_.begin(); t != textures_.end(); t++) {
		cuParamSetTexRef(_f, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

    // TODO: add support for events
    cl_int ret = clEnqueueNDRangeKernel(stream_, f_, workDim_, globalWorkOffset_, globalWorkSize_, localWorkSize_, 0, NULL, NULL);

    return Accelerator::error(ret);
}

}}
