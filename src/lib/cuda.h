#ifndef __CUDA_H_
#define __CUDA_H_

#include <driver_types.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaLaunch(const char *);
cudaError_t cudaThreadSynchronize(void);

#ifdef __cplusplus
};
#endif

#endif
