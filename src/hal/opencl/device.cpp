#include "hal/types.h"

#include "util/Logger.h"

#include "device.h"

namespace __impl { namespace hal { namespace opencl {

device::device(cl_device_id openclDeviceId,
               cl_platform_id openclPlatformId,
               coherence_domain &coherenceDomain) :
    Parent(coherenceDomain),
    openclDeviceId_(openclDeviceId),
    openclPlatformId_(openclPlatformId)
{
    // Memory size
    {
        cl_int ret = CL_SUCCESS;
        cl_ulong val = 0;

        TRACE(GLOBAL, "Creating OPENCL accelerator");
        ret = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(val), &val, NULL);
        CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);

        memorySize_ = val;
    }

    // OpenCL version
    {
        openclVersion_ = util::get_opencl_version(openclPlatformId);
    }

    // Uses integrated memory
    {
        cl_bool val = CL_FALSE;
        cl_int ret = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_HOST_UNIFIED_MEMORY,
                sizeof(val), NULL, NULL);
        ASSERTION(ret == CL_SUCCESS, "Error querying device for unified memory");
        integrated_ = (val == CL_TRUE);
    }
}

aspace_t
device::create_address_space(const Parent::SetSiblings &siblings)
{
    cl_int ret;
    cl_context ctx;

    cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) openclPlatformId_,
        0
    };

    // Checks
    {
        ASSERTION(siblings.size() >= 1, "Siblings' set is empty");
        ASSERTION(siblings.find(this) != siblings.end(), "Siblings does not contain the current device");

        cl_device_id *clDevices = new cl_device_id[siblings.size()];
        int i = 0;

        for (SetSiblings::const_iterator it = siblings.begin(); it != siblings.end(); it++) {
            ASSERTION(&(*it)->get_coherence_domain() == &coherenceDomain_, "Coherence domain do not match");
            clDevices[i++] = reinterpret_cast<const device *>(*it)->openclDeviceId_;
        }

        ctx = clCreateContext(prop, siblings.size(), clDevices, NULL, NULL, &ret);
        ASSERTION(ret == CL_SUCCESS);

        return aspace_t(ctx, *this);
    }
}

stream_t
device::create_stream(aspace_t &aspace)
{
    cl_command_queue stream;
    cl_int error;
    cl_command_queue_properties prop = 0;
#if defined(USE_TRACE)
    prop |= CL_QUEUE_PROFILING_ENABLE;
#endif
    stream = clCreateCommandQueue(aspace(), openclDeviceId_, prop, &error);
    CFATAL(error == CL_SUCCESS, "Unable to create OpenCL stream");
    return stream_t(stream, aspace);
}

event_t
device::copy(accptr_t dst, hostptr_t src, size_t count, stream_t &stream)
{
    event_t event(stream);

    cl_int err = clEnqueueWriteBuffer(stream(), dst.get(), CL_TRUE, dst.offset(), count, src, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

event_t
device::copy(hostptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    event_t event(stream);

    cl_int err = clEnqueueReadBuffer(stream(), src.get(), CL_TRUE, src.offset(), count, dst, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

event_t
device::copy(accptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    event_t event(stream);

    cl_int err = clEnqueueCopyBuffer(stream(),
                                     src.get(), dst.get(),
                                     src.offset(), dst.offset(),
                                     count, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    err = clWaitForEvents(1, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

async_event_t
device::copy_async(accptr_t dst, hostptr_t src, size_t count, stream_t &stream)
{
    async_event_t event(stream);

    cl_int err = clEnqueueWriteBuffer(stream(), dst.get(), CL_FALSE, dst.offset(), count, src, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

async_event_t
device::copy_async(hostptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    async_event_t event(stream);

    cl_int err = clEnqueueReadBuffer(stream(), src.get(), CL_FALSE, src.offset(), count, dst, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

async_event_t
device::copy_async(accptr_t dst, accptr_t src, size_t count, stream_t &stream)
{
    async_event_t event(stream);

    cl_int err = clEnqueueCopyBuffer(stream(),
                                     src.get(), dst.get(),
                                     src.offset(), dst.offset(),
                                     count, 0, NULL, &event());
    ASSERTION(err == CL_SUCCESS);

    return event;
}

gmacError_t
device::sync(async_event_t &event)
{
    cl_int res;

    res = clWaitForEvents(1, &event());

    return opencl::error(res);
}

gmacError_t
device::sync(stream_t &stream)
{
    cl_int res;

    res = clFinish(stream());

    return opencl::error(res);
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
