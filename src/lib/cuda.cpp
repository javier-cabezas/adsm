#define NATIVE
#include "gmac.h"

#include "cuda.h"

cudaError_t cudaLaunch(const char *name)
{
	return gmacLaunch(name);
}

cudaError_t cudaThreadSynchronize()
{
	return gmacThreadSynchronize();
}
