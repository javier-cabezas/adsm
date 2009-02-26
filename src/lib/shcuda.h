#ifndef __SHCUDA_H_
#define __SHCUDA_H_

#ifndef NATIVE
#define cudaMalloc(...) shCudaMalloc(__VA_ARGS__)
#define cudaFree(...) shCudaFree(__VA_ARGS__)
#define cudaMallocPitch(...) shCudaMallocPitch(__VA_ARGS__)
#define cudaLaunch(...) shCudaLaunch(__VA_ARGS__)
#define cudaThreadSynchronize(...) shCudaThreadSynchronize(__VA_ARGS__)
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

cudaError_t shCudaMalloc(void **, size_t);
cudaError_t shCudaFree(void *);
cudaError_t shCudaLaunch(const char *);
cudaError_t shCudaThreadSynchronize(void);


#ifdef __cplusplus
};
#endif

#endif
