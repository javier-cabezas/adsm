#include "GPU.h"

#include <debug.h>

namespace gmac {

gmacError_t GPU::error(CUresult ret)
{
	gmacError_t _error;
	switch(ret) {
		case CUDA_SUCCESS: _error = gmacSuccess; break;
		case CUDA_ERROR_OUT_OF_MEMORY:
			_error = gmacErrorMemoryAllocation; break;
		case CUDA_ERROR_LAUNCH_FAILED:
			_error = gmacErrorLaunchFailure; break;
		default: _error = gmacErrorUnknown; break;
	}
	return _error;
}

gmacError_t GPU::memset(void *addr, int i, size_t n)
{
	CUresult ret = CUDA_SUCCESS;
	unsigned char c = i & 0xff;
	if((n % 4) == 0) {
		unsigned m = c | (c << 8);
		m |= (m << 16);
		ret = cuMemsetD32(gpuAddr(addr), m, n / 4);
	}
	else if((n % 2) == 0) {
		unsigned short s = c | (c << 8);
		ret = cuMemsetD16(gpuAddr(addr), s, n / 2);
	}
	else {
		ret = cuMemsetD8(gpuAddr(addr), c, n);
	}
	return error(ret);
}

gmacError_t GPU::launch(dim3 Dg, dim3 Db, CUfunction f)
{
	CUresult ret = CUDA_SUCCESS;
	if((ret = cuFuncSetBlockShape(f, Db.x, Db.y, Db.z)) != CUDA_SUCCESS)
		return error(ret);

	if((ret = cuLaunchGrid(f, Dg.x, Dg.y)) != CUDA_SUCCESS)
		return error(ret);

	return error(ret);
}

};
