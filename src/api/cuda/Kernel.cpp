#include "Kernel.h"
#include "Module.h"
#include "Mode.h"
#include "Accelerator.h"

#include "trace/Tracer.h"

namespace __impl { namespace cuda {

Kernel::Kernel(const core::KernelDescriptor & k, CUmodule mod) :
    core::Kernel(k)
{
    CUresult ret = cuModuleGetFunction(&_f, mod, name_);
    //! \todo Calculate this dynamically
#if CUDA_VERSION >= 3000 && LINUX
    ret = cuFuncSetCacheConfig(_f, CU_FUNC_CACHE_PREFER_L1);
    ASSERTION(ret == CUDA_SUCCESS);
#endif
    ASSERTION(ret == CUDA_SUCCESS);
}

core::KernelLaunch *
Kernel::launch(core::KernelConfig & _c)
{
    KernelConfig & c = static_cast<KernelConfig &>(_c);

    KernelLaunch * l = new cuda::KernelLaunch(*this, c);
    return l;
}

KernelConfig::KernelConfig(const KernelConfig & c) :
    core::KernelConfig(c),
    _grid(c._grid),
    _block(c._block),
    _shared(c._shared),
    _stream(c._stream)
{
}

KernelConfig::KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t /*tokens*/) :
    core::KernelConfig(),
    _grid(grid),
    _block(block),
    _shared(shared),
    _stream(NULL)
{
}

KernelLaunch::KernelLaunch(const Kernel & k, const KernelConfig & c) :
    core::KernelLaunch(),
    cuda::KernelConfig(c),
    _kernel(k),
    _f(k._f)
{
}

KernelLaunch &KernelLaunch::operator =(const KernelLaunch &)
{
    FATAL("Assigment of kernel launch is not supported");
    return *this;
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    CUresult ret = cuParamSetv(_f, 0, argsArray(), argsSize());
    CFATAL(ret == CUDA_SUCCESS, "CUDA Error setting parameters: %d", ret);
    ret = cuParamSetSize(_f, argsSize());
	ASSERTION(ret == CUDA_SUCCESS);

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = textures_.begin(); t != textures_.end(); t++) {
		cuParamSetTexRef(_f, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(_f, (unsigned int)shared())) != CUDA_SUCCESS) {
        goto exit;
	}

	if((ret = cuFuncSetBlockShape(_f, block().x, block().y, block().z))
			!= CUDA_SUCCESS) {
        goto exit;
	}

	ret = cuLaunchGridAsync(_f, grid().x, grid().y, _stream);

exit:
    return Accelerator::error(ret);
}

}}
