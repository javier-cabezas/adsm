#include "api.h"

#include <gmac/gmac.h>

#include <cuda_runtime.h>
#include <cuda.h>

DECL_SYM(__gmacLaunch);

static void __attribute__((constructor)) gmacCudaInit(void)
{
	LOAD_SYM(__gmacLaunch, cudaLaunch);
}

cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

