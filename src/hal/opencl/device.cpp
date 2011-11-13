#include "util/Logger.h"

#include "hal/types.h"
#include "device.h"

namespace __impl { namespace hal { namespace opencl {

cl_device_id *
platform::get_cl_device_array()
{
    cl_device_id *deviceIds = new cl_device_id[devices_.size()];

    unsigned i = 0;
    for (std::list<device *>::iterator it  = devices_.begin();
            it != devices_.end();
            it++) {
        deviceIds[i++] = (*it)->openclDeviceId_;
    }
    return deviceIds;
}


device::device(platform &p,
               cl_device_id openclDeviceId,
               coherence_domain &coherenceDomain) :
    Parent(coherenceDomain),
    platform_(p),
    openclDeviceId_(openclDeviceId),
    context_(p.get_context())
{
    // Memory size
    {
        cl_int ret = CL_SUCCESS;
        cl_ulong val = 0;

        TRACE(GLOBAL, "Creating OpenCL accelerator");
        ret = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(val), &val, NULL);
        CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);

        memorySize_ = val;
    }

    // OpenCL version
    {
        openclVersion_ = helper::get_opencl_version(platform_.get_cl_platform_id());
    }

    // Uses integrated memory
    {
        cl_bool val = CL_FALSE;
        cl_int ret = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_HOST_UNIFIED_MEMORY,
                sizeof(val), &val, NULL);
        ASSERTION(ret == CL_SUCCESS, "Error querying device for unified memory");
        integrated_ = (val == CL_TRUE);
    }

    p.add_device(*this);
}

context_t *
device::create_context(const SetSiblings &siblings, gmacError_t &err)
{
    context_t *ret = NULL;

#if 0
    cl_device_id *devices = platform_.get_cl_device_array();
    unsigned ndevices = platform_.get_ndevices();

    cl_context_properties prop[] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties) platform_.get_cl_platform_id(),
        0
    };

    //ctx = clCreateContext(prop, ndevices, devices, NULL, NULL, &res);
#endif

    ret = new context_t(platform_.get_context(), *this);

#if 0
    // Checks
    {
        ASSERTION(siblings.size() >= 1, "Siblings' set is empty");
        ASSERTION(siblings.find(this) != siblings.end(), "Siblings does not contain the current device");

        cl_device_id *clDevices = new cl_device_id[siblings.size() + 1];
        clDevices[0] = openclDeviceId_;

        unsigned i = 1;

        for (SetSiblings::const_iterator it = siblings.begin(); it != siblings.end(); it++) {
            ASSERTION(&(*it)->get_coherence_domain() == &coherenceDomain_, "Coherence domain do not match");
            clDevices[i++] = reinterpret_cast<const device *>(*it)->openclDeviceId_;
        }

        ctx = clCreateContext(prop, siblings.size(), clDevices, NULL, NULL, &ret);
        ASSERTION(ret == CL_SUCCESS);

        return new context_t(ctx, *this);
    }

    if (devices != NULL) {
        delete []devices;
    }
#endif

    return ret;
}

gmacError_t
device::destroy_context(context_t &context)
{
    cl_int ret = clReleaseContext(context());

    return error(ret);
}

stream_t *
device::create_stream(context_t &context)
{
    cl_command_queue stream;
    cl_int error;
    cl_command_queue_properties prop = 0;
#if defined(USE_TRACE)
    prop |= CL_QUEUE_PROFILING_ENABLE;
#endif
    stream = clCreateCommandQueue(context(), openclDeviceId_, prop, &error);
    CFATAL(error == CL_SUCCESS, "Unable to create OpenCL stream");
    return new stream_t(stream, context);
}

gmacError_t
device::destroy_stream(stream_t &stream)
{
    cl_int ret = clReleaseCommandQueue(stream());

    return error(ret);
}

size_t
device::get_total_memory() const
{
    FATAL("Not implemented");
    return 0;
}

size_t
device::get_free_memory() const
{
    FATAL("Not implemented");
    return 0;
}

bool
device::has_direct_copy(const Parent &_dev) const
{
    const device &dev = reinterpret_cast<const device &>(_dev);
#if 0
    int canAccess;
    CUresult ret = cuDeviceCanAccessPeer(&canAccess, cudaDevice_, dev.cudaDevice_);
    ASSERTION(ret == CUDA_SUCCESS, "Error querying devices");

    return canAccess == 1;
#endif
    return &get_platform() == &dev.get_platform();
}


}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
