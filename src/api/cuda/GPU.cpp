#include "GPU.h"

#include <debug.h>

namespace gmac {

gmacError_t GPU::error(cudaError_t ret)
{
	gmacError_t _error;
	switch(ret) {
		case cudaSuccess: _error = gmacSuccess; break;
		case cudaErrorMemoryAllocation:
			_error = gmacErrorMemoryAllocation; break;
		case cudaErrorLaunchFailure:
			_error = gmacErrorLaunchFailure; break;
		default: _error = gmacErrorUnknown; break;
	}
	return _error;
}

};
