#include "Context.h"

namespace gmac { namespace gpu {

#define ERROR(r, err) case r: error = err; break

gmacError_t
Context::error(CUresult r)
{
	gmacError_t error = gmacSuccess;
	switch(r) {
		ERROR(CUDA_SUCCESS, gmacSuccess);
		ERROR(CUDA_ERROR_INVALID_VALUE, gmacErrorInvalidValue);
		ERROR(CUDA_ERROR_OUT_OF_MEMORY, gmacErrorMemoryAllocation);
		ERROR(CUDA_ERROR_NOT_INITIALIZED, gmacErrorNotReady);
		ERROR(CUDA_ERROR_DEINITIALIZED, gmacErrorNotReady);
		ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoDevice);
		ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidDevice);
		ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidDeviceFunction);
		ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
		ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidDeviceFunction);	
		ERROR(CUDA_ERROR_ALREADY_ACQUIRED, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_FILE_NOT_FOUND, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_INVALID_HANDLE, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_NOT_FOUND, gmacErrorApiFailureBase);
		ERROR(CUDA_ERROR_NOT_READY, gmacErrorNotReady);
		ERROR(CUDA_ERROR_LAUNCH_FAILED, gmacErrorLaunchFailure);	
		ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, gmacErrorLaunchFailure);
		ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, gmacErrorLaunchFailure);
		ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, gmacErrorLaunchFailure);
		ERROR(CUDA_ERROR_UNKNOWN, gmacErrorUnknown);
		default: error = gmacErrorUnknown;
	}
    _error = error;
	return error;
}

}}
