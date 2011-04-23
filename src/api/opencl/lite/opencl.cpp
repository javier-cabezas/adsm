#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

#include "util/Logger.h"

#include <CL/cl.h>

SYM(cl_context, __opencl_clCreateContext,
        const cl_context_properties *,
        cl_uint,
        const cl_device_id *,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

SYM(cl_context, __opencl_clCreateContextFromType,
        const cl_context_properties,
        cl_device_type,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

SYM(cl_int, __opencl_clRetainContext, cl_context);

SYM(cl_int, __opencl_clCReleaseContext, cl_context);

SYM(cl_int, __opencl_clEnqueueNDRangeKernel,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const size_t *,
        const size_t *,
        const size_t *,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clEnqueueTask,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clEnqueueNativeKernel,
        cl_command_queue,
        void (*)(void *),
        void *,
        size_t,
        cl_uint,
        const cl_mem *,
        const void **,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clFinish, cl_command_queue);

static void openclInit()
{
    LOAD_SYM(__opencl_clCreateContext, clCreateContext);
}

#ifdef __cplusplus
extern "C" {
#endif

cl_context SYMBOL(clCreateContext)(
        const cl_context_properties *properties,
        cl_uint num_devices,
        const cl_device_id *devices,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    if(errcode_ret != CL_SUCCESS) return ret;

    return ret;
}


#ifdef __cplusplus
}
#endif
