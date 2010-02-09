#include "Context.h"

namespace gmac { namespace gpu {

#define ERROR(r, err) case r: error = err; break

gmacError_t Context::error(cudaError_t e)
{
	gmacError_t error = gmacSuccess;
	switch(e) {
		ERROR(cudaSuccess, gmacSuccess);
		ERROR(cudaErrorMemoryAllocation, gmacErrorMemoryAllocation);
		ERROR(cudaErrorLaunchFailure, gmacErrorLaunchFailure);
		ERROR(cudaErrorNotReady, gmacErrorNotReady);
		ERROR(cudaErrorNoDevice, gmacErrorNoDevice);	
		ERROR(cudaErrorInvalidValue, gmacErrorInvalidValue);
		ERROR(cudaErrorInvalidDevice, gmacErrorInvalidDevice);
		ERROR(cudaErrorInvalidDeviceFunction, gmacErrorInvalidDeviceFunction);
		ERROR(cudaErrorApiFailureBase, gmacErrorApiFailureBase);
		default: error = gmacErrorUnknown;
	}
	return error;
}

}}

