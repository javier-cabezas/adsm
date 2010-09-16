#include "Kernel.h"
#include "Module.h"
#include "Mode.h"
#include "Accelerator.h"

#include <trace/Thread.h>

#include <cuda_runtime_api.h>

namespace gmac { namespace cuda {

Kernel::Kernel(const gmac::KernelDescriptor & k, CUmodule mod) :
    gmac::Kernel(k)
{
    CUresult ret = cuModuleGetFunction(&_f, mod, _name);
    //! \todo Calculate this dynamically
#if CUDART_VERSION >= 3000 && LINUX
    cuFuncSetCacheConfig(_f, CU_FUNC_CACHE_PREFER_L1);
#endif
    assertion(ret == CUDA_SUCCESS);
}

gmac::KernelLaunch *
Kernel::launch(gmac::KernelConfig & _c)
{
    KernelConfig & c = static_cast<KernelConfig &>(_c);

    KernelLaunch * l = new KernelLaunch(*this, c);
    return l;
}

KernelConfig::KernelConfig(const KernelConfig & c) :
    gmac::KernelConfig(c),
    _grid(c._grid),
    _block(c._block),
    _shared(c._shared),
    _stream(c._stream)
{
}

KernelConfig::KernelConfig(dim3 grid, dim3 block, size_t shared, cudaStream_t tokens) :
    gmac::KernelConfig(),
    _grid(grid),
    _block(block),
    _shared(shared),
    _stream(NULL)
{
}

KernelLaunch::KernelLaunch(const Kernel & k, const KernelConfig & c) :
    gmac::KernelLaunch(),
    KernelConfig(c),
    _kernel(k),
    _f(k._f)
{
}

gmacError_t
KernelLaunch::execute()
{
	// Set-up parameters
    CUresult ret = cuParamSetv(_f, 0, argsArray(), argsSize());
    CFatal(ret == CUDA_SUCCESS, "CUDA Error setting parameters: %d", ret);
    ret = cuParamSetSize(_f, argsSize());
	assertion(ret == CUDA_SUCCESS);

#if 0
	// Set-up textures
	Textures::const_iterator t;
	for(t = _textures.begin(); t != _textures.end(); t++) {
		cuParamSetTexRef(_f, CU_PARAM_TR_DEFAULT, *(*t));
	}
#endif

	// Set-up shared size
	if((ret = cuFuncSetSharedSize(_f, shared())) != CUDA_SUCCESS) {
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
