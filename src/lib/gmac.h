#ifndef __SHCUDA_H_
#define __SHCUDA_H_

#ifndef NATIVE
#define cudaMalloc(...) gmacMalloc(__VA_ARGS__)
#define cudaFree(...) gmacFree(__VA_ARGS__)
#define cudaMallocPitch(...) gmacMallocPitch(__VA_ARGS__)
#define cudaLaunch(...) gmacLaunch(__VA_ARGS__)
#define cudaThreadSynchronize(...) gmacThreadSynchronize(__VA_ARGS__)
#else
#define cudaMalloc(...) cudaMalloc(__VA_ARGS__)
#define cudaFree(...) cudaFree(__VA_ARGS__)
#define cudaMallocPitch(...) cudaMallocPitch(__VA_ARGS__)
#define cudaLaunch(...) cudaLaunch(__VA_ARGS__)
#define cudaThreadSynchronize(...) cudaThreadSynchronize(__VA_ARGS__)
#endif

#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t gmacMalloc(void **, size_t);
cudaError_t gmacFree(void *);
cudaError_t gmacLaunch(const char *);
cudaError_t gmacThreadSynchronize(void);


#ifdef __cplusplus
};
#endif

#endif
