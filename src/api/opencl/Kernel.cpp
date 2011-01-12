#include "Kernel.h"
#include "Module.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace cuda {

Kernel::Kernel(const core::KernelDescriptor & k, cl_program program) :
    core::Kernel(k)
{
    cl_int ret;
    f_ = clCreateKernel(program, name_, &ret);
    ASSERTION(ret == CL_SUCCESS);
}

core::KernelLaunch *
Kernel::launch(core::KernelConfig & _c)
{
    KernelConfig & c = static_cast<KernelConfig &>(_c);

    KernelLaunch * l = new opencl::KernelLaunch(*this, c);
    return l;
}

KernelConfig::KernelConfig(cl_uint work_dim, size_t *global_work_offset, size_t *global_work_size, size_t *local_work_size, cl_command_queue stream)
    core::KernelConfig(),
    work_dim_(work_dim_),
    stream_(stream)
{
    global_work_offset_ = new size_t[work_dim];
    global_work_size_ = new size_t[work_dim];
    local_work_size_ = new size_t[work_dim];

    for (unsigned i = 0; i < work_dim; i++) {
        global_work_offset_[i] = global_work_offset[i];
        global_work_size_[i] = global_work_size[i];
        local_work_size_[i] = local_work_size[i];
    }
}

KernelConfig::~KernelConfig()
{
    delete [] global_work_offset_;
    delete [] global_work_size_;
    delete [] local_work_size_;
}

KernelLaunch::KernelLaunch(const Kernel & k, const KernelConfig & c) :
    core::KernelLaunch(),
    cuda::KernelConfig(c),
    f_(k._f)
{
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    unsigned i = 0;
    for (std::vector<core::Argument>::const_iterator it = begin(); it != end(); it++) {
        cl_int ret = clSetKernelArg(f_, i, it->size(), it->ptr());
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
    cl_int ret = clEnqueueNDRangeKernel(stream_, f_, work_dim_, global_work_offset_,
            global_work_size_, local_work_size_, 0, NULL, NULL)

exit:
    return Accelerator::error(ret);
}

}}
