#include "Kernel.h"
#include "Module.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace cuda {

Kernel::Kernel(const core::KernelDescriptor & k, CUmodule mod) :
    gmac::core::Kernel(k)
{
    CUresult ret = cuModuleGetFunction(&f_, mod, name_);
    ASSERTION(ret == CUDA_SUCCESS);
    //! \todo Calculate this dynamically
#if CUDA_VERSION >= 3000 && LINUX
    ret = cuFuncSetCacheConfig(f_, CU_FUNC_CACHE_PREFER_L1);
    ASSERTION(ret == CUDA_SUCCESS);
#endif
    ASSERTION(ret == CUDA_SUCCESS);
}

KernelConfig::KernelConfig(const KernelConfig & c) :
    argsSize_(0),
    grid_(c.grid_),
    block_(c.block_),
    shared_(c.shared_),
    stream_(c.stream_)
{
    ArgsVector::const_iterator it;
    for (it = c.begin(); it != c.end(); it++) {
        pushArgument(it->ptr(), it->size(), it->offset());
    }
}


KernelConfig::KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t /*tokens*/, CUstream stream) :
    argsSize_(0),
    grid_(grid),
    block_(block),
    shared_(shared),
    stream_(stream)
{
}


KernelLaunch::KernelLaunch(core::Mode &mode, const Kernel & k, const KernelConfig & c) :
    core::KernelLaunch(mode),
    KernelConfig(c),
    kernel_(k),
    f_(k.f_)
{
}

KernelLaunch::~KernelLaunch()
{
    cuEventDestroy(start_);
    cuEventDestroy(end_);
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    CUresult ret = cuParamSetv(f_, 0, argsArray(), unsigned(argsSize()));
    CFATAL(ret == CUDA_SUCCESS, "CUDA Error setting parameters: %d", ret);
    ret = cuParamSetSize(f_, unsigned(argsSize()));
	CFATAL(ret == CUDA_SUCCESS);

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = textures_.begin(); t != textures_.end(); t++) {
		cuParamSetTexRef(f_, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(f_, (unsigned int)shared())) != CUDA_SUCCESS) {
        goto exit;
	}

	if((ret = cuFuncSetBlockShape(f_, block().x, block().y, block().z))
			!= CUDA_SUCCESS) {
        goto exit;
	}

    if (cuEventCreate(&start_, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
        goto exit;
    }
    if (cuEventCreate(&end_, CU_EVENT_DEFAULT) != CUDA_SUCCESS) {
        goto exit;
    }

    if (cuEventRecord(start_, stream_) != CUDA_SUCCESS) goto exit;

	ret = cuLaunchGridAsync(f_, grid().x, grid().y, stream_);

    if (ret != CUDA_SUCCESS) goto exit;

    cuEventRecord(end_, stream_);

exit:
    return Accelerator::error(ret);
}

}}
