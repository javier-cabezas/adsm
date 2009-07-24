#include "api.h"
#include <os/loader.h>

#include <cuda_runtime.h>
#include <cuda.h>


SYM(gmacError_t, __gmacLaunch, const char *);

static void __attribute__((constructor(101))) gmacCudaInit(void)
{
	LOAD_SYM(__gmacLaunch, cudaLaunch);
}

#ifdef __cplusplus
extern "C" {
#endif


extern gmacError_t gmacLaunch(const char *);
cudaError_t cudaLaunch(const char *symbol)
{
	return gmacLaunch(symbol);
}

#ifdef __cplusplus
}
#endif


