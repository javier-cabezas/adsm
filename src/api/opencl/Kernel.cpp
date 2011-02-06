#include "Kernel.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace opencl {

KernelConfig::KernelConfig(cl_uint workDim, size_t *globalWorkOffset, size_t *globalWorkSize, size_t *localWorkSize, cl_command_queue stream) :
    argsSize_(0),
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

KernelConfig::KernelConfig(const KernelConfig &config) :
    argsSize_(0),
    workDim_(config.workDim_),
    globalWorkOffset_(NULL),
    globalWorkSize_(NULL),
    localWorkSize_(NULL),
    stream_(config.stream_)
{
    if(config.globalWorkOffset_) globalWorkOffset_ = new size_t[workDim_];
    if(config.globalWorkSize_) globalWorkSize_ = new size_t[workDim_];
    if(config.localWorkSize_) localWorkSize_ = new size_t[workDim_];

    for(unsigned i = 0; i < workDim_; i++) {
        if(globalWorkOffset_) globalWorkOffset_[i] = config.globalWorkOffset_[i];
        if(globalWorkSize_) globalWorkSize_[i] = config.globalWorkSize_[i];
        if(localWorkSize_) localWorkSize_[i] = config.localWorkSize_[i];
    }

    ArgsList::const_iterator it;
    for (it = config.begin(); it != config.end(); it++) {
        setArgument(it->ptr(), it->size(), it->index());
    }
}

KernelConfig &KernelConfig::operator=(const KernelConfig &config)
{
    if(this == &config) return *this;

    workDim_ = config.workDim_;
    globalWorkOffset_ = NULL;
    globalWorkSize_ = NULL;
    localWorkSize_ = NULL;
    stream_ = config.stream_;

    if(config.globalWorkOffset_) globalWorkOffset_ = new size_t[workDim_];
    if(config.globalWorkSize_) globalWorkSize_ = new size_t[workDim_];
    if(config.localWorkSize_) localWorkSize_ = new size_t[workDim_];

    for(unsigned i = 0; i < workDim_; i++) {
        if(globalWorkOffset_) globalWorkOffset_[i] = config.globalWorkOffset_[i];
        if(globalWorkSize_) globalWorkSize_[i] = config.globalWorkSize_[i];
        if(localWorkSize_) localWorkSize_[i] = config.localWorkSize_[i];
    }

    argsSize_ = 0;
    clear();

    ArgsList::const_iterator it;
    for (it = config.begin(); it != config.end(); it++) {
        setArgument(it->ptr(), it->size(), it->index());
    }

    return *this;
}


gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    for (ArgsList::const_iterator it = begin(); it != end(); it++) {
        TRACE(LOCAL, "Setting param %u @ %p ("FMT_SIZE")", it->index(), it->ptr(), it->size());
        cl_int ret = clSetKernelArg(f_, it->index(), it->size(), it->ptr());
        CFATAL(ret == CL_SUCCESS, "OpenCL Error setting parameters: %d", ret);
    }

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = textures_.begin(); t != textures_.end(); t++) {
		cuParamSetTexRef(_f, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

    // TODO: add support for events
    TRACE(LOCAL, "Launch kernel %u %zd %zd @ %p", workDim_, globalWorkSize_[0], localWorkSize_[0], stream_);
    cl_int ret = clEnqueueNDRangeKernel(stream_, f_, workDim_, globalWorkOffset_, globalWorkSize_, localWorkSize_, 0, NULL, NULL);

    return Accelerator::error(ret);
}

}}
