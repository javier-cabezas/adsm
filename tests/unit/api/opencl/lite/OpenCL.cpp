#include "OpenCL.h"

#include "CL/cl.h"

bool CreateOpenCLContext(cl_device_id &device, cl_context &context)
{
    cl_int error_code;
    cl_uint num_devices;
    cl_platform_id platform;
    
    error_code = clGetPlatformIDs(1, &platform, NULL);
    if(error_code != CL_SUCCESS) return false;

    cl_context_properties context_properties[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    context = clCreateContextFromType(context_properties, CL_DEVICE_TYPE_GPU, NULL, NULL, &error_code);
    if(error_code != CL_SUCCESS) return false;

    error_code = clGetContextInfo(context, CL_CONTEXT_NUM_DEVICES,
            sizeof(cl_uint), &num_devices, NULL);
    if(error_code != CL_SUCCESS || num_devices != 1) goto context_cleanup;

    error_code = clGetContextInfo(context, CL_CONTEXT_DEVICES,
            sizeof(cl_device_id), &device, NULL);
    if(error_code != CL_SUCCESS) goto context_cleanup;
    
    return true;

context_cleanup:
    clReleaseContext(context);
    return false;
}
