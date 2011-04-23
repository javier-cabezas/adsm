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
        const cl_context_properties *,
        cl_device_type,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

SYM(cl_int, __opencl_clRetainContext, cl_context);

SYM(cl_int, __opencl_clReleaseContext, cl_context);

SYM(cl_command_queue, __opencl_clCreateCommandQueue,
        cl_context,
        cl_device_id,
        cl_command_queue_properties,
        cl_int *);

SYM(cl_int, __opencl_clRetainCommandQueue, cl_command_queue);

SYM(cl_int, __opencl_clReleaseCommandQueue, cl_command_queue);

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
    LOAD_SYM(__opencl_clCreateContextFromType, clCreateContextFromType);
    LOAD_SYM(__opencl_clRetainContext, clRetainContext);
    LOAD_SYM(__opencl_clReleaseContext, clReleaseContext);

    LOAD_SYM(__opencl_clCreateCommandQueue, clCreateCommandQueue);
    LOAD_SYM(__opencl_clRetainCommandQueue, clRetainCommandQueue);
    LOAD_SYM(__opencl_clReleaseCommandQueue, clReleaseCommandQueue);
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
    if(*errcode_ret != CL_SUCCESS) return ret;

    return ret;
}

cl_context SYMBOL(clCreateContextFromType)(
        const cl_context_properties *properties,
        cl_device_type device_type,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
    if(*errcode_ret != CL_SUCCESS) return ret;
    return ret;
}
        
cl_int SYMBOL(clRetainContext)(cl_context context)
{
    if(__opencl_clRetainContext == NULL) openclInit();
    cl_int ret = __opencl_clRetainContext(context);
    if(ret != CL_SUCCESS) return ret;
    return ret;
}

cl_int SYMBOL(clReleaseContext)(cl_context context)
{
    if(__opencl_clReleaseContext == NULL) openclInit();
    cl_int ret = __opencl_clReleaseContext(context);
    if(ret != CL_SUCCESS) return ret;
    return ret;
}

cl_command_queue SYMBOL(clCreateCommandQueue)(
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateCommandQueue == NULL) openclInit();
    cl_command_queue ret = __opencl_clCreateCommandQueue(context, device, properties,  errcode_ret);
    if(*errcode_ret != CL_SUCCESS) return ret;

    return ret;
}

cl_int SYMBOL(clRetainCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clRetainCommandQueue == NULL) openclInit();
    cl_int ret = __opencl_clRetainCommandQueue(command_queue);
    if(ret != CL_SUCCESS) return ret;

    return ret;
}

cl_int SYMBOL(clReleaseCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clReleaseCommandQueue == NULL) openclInit();
    cl_int ret = __opencl_clRetainCommandQueue(command_queue);
    if(ret != CL_SUCCESS) return ret;

    return ret;
}

static void CONSTRUCTOR init()
{
    fprintf(stderr,"Init GMAC/Lite\n");
}

#ifdef __cplusplus
}
#endif
