#include "api.h"
#include <os/loader.h>

#include <cuda_runtime.h>
#include <cuda.h>


SYM(gmacError_t, __gmacLaunch, const char *);
DECL_SYM(__gmacLaunch);

static void __attribute__((constructor)) gmacCudaInit(void)
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


