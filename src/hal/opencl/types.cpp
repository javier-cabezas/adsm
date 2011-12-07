#include "config/common.h"

#include "hal/opencl/helper/opencl_helper.h"
#include "hal/opencl/kernels/kernels.h"

#include "util/file.h"

#include "coherence_domain.h"
#include "device.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; break

namespace __impl { namespace hal { namespace opencl { 

map_platform_repository Modules_("map_platform_repository");

gmacError_t
compile_embedded_code(std::list<opencl::device *> devices)
{
    return gmacErrorFeatureNotSupported;
}

static gmacError_t
register_gmac_kernels(std::list<opencl::platform *> platforms)
{
    for (unsigned i = 0; i < 1; i++) {
        /* module_descriptor *descriptor = */new module_descriptor(KernelsGmac_[i], "");
    }

    return gmacSuccess;
}

}}}

namespace __impl { namespace hal {

gmacError_t
init_platform()
{
    static bool initialized = false;
    gmacError_t ret = gmacSuccess;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL platform initialization");
    }

    return ret;
}

std::list<opencl::device *>
init_devices()
{
    static bool initialized = false;

    if (initialized == false) {
        initialized = true;
    } else {
        FATAL("Double HAL device initialization");
    }

    std::list<opencl::device *> devices;
    std::list<opencl::platform *> platforms;

    TRACE(GLOBAL, "Initializing OpenCL API");
    cl_uint platformSize = 0;
    cl_int ret = CL_SUCCESS;
    ret = clGetPlatformIDs(0, NULL, &platformSize);
    CFATAL(ret == CL_SUCCESS);
    if(platformSize == 0) return devices;   
    cl_platform_id * platform_ids = new cl_platform_id[platformSize];
    ret = clGetPlatformIDs(platformSize, platform_ids, NULL);
    CFATAL(ret == CL_SUCCESS);
    MESSAGE("%d OpenCL platforms found", platformSize);

    for (unsigned i = 0; i < platformSize; i++) {
        MESSAGE("Platform [%u/%u]: %s", i + 1, platformSize, opencl::helper::get_platform_name(platform_ids[i]).c_str());
        cl_uint deviceSize = 0;
        ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU,
                             0, NULL, &deviceSize);
        ASSERTION(ret == CL_SUCCESS);
	    if(deviceSize == 0) continue;
        cl_device_id *deviceIds = new cl_device_id[deviceSize];
        ret = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_GPU,
                             deviceSize, deviceIds, NULL);
        ASSERTION(ret == CL_SUCCESS);
        MESSAGE("... found %u OpenCL devices", deviceSize, i);

#if 0
        opencl::helper::opencl_version clVersion = opencl::helper::get_opencl_version(platform_ids[i]);
#endif

        cl_context ctx;
        opencl::platform *plat;

        cl_context_properties prop[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform_ids[i], 0 };

        ctx = clCreateContext(prop, deviceSize, deviceIds, NULL, NULL, &ret);
        CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL context %d", ret);

        TRACE(GLOBAL, "cl_context %p for platform: %s", ctx, opencl::helper::get_platform_name(platform_ids[i]).c_str());

        plat = new opencl::platform(platform_ids[i], ctx);

        for (unsigned j = 0; j < deviceSize; j++) {
            MESSAGE("Device [%u/%u]: %s", j + 1, deviceSize, opencl::helper::get_device_name(deviceIds[j]).c_str());

            // Let's assume that this is not important; TODO: actually deal with this case
            //CFATAL(util::getDeviceVendor(devices[j]) == util::getPlatformVendor(platform_ids[i]), "Not handled case");
            opencl::device *device = NULL;

            switch (opencl::helper::get_platform(platform_ids[i])) {
                case opencl::helper::PLATFORM_AMD:
                    if (opencl::helper::is_device_amd_fusion(deviceIds[j])) {
                        device = new opencl::device(*plat, deviceIds[j], *new opencl::coherence_domain());
                    } else {
                        device = new opencl::device(*plat, deviceIds[j], *new opencl::coherence_domain());
                    }
                    break;
                case opencl::helper::PLATFORM_APPLE:
                case opencl::helper::PLATFORM_NVIDIA:
                    device = new opencl::device(*plat, deviceIds[j], *new opencl::coherence_domain());
                    break;
                case opencl::helper::PLATFORM_INTEL:
                case opencl::helper::PLATFORM_UNKNOWN:
                    FATAL("Platform not supported\n");
            }
            devices.push_back(device);
        }
        delete[] deviceIds;

        platforms.push_back(plat);
    }
    delete[] platform_ids;
    initialized = true;

    opencl::register_gmac_kernels(platforms);
    opencl::compile_embedded_code(devices);

    return devices;
}

}}

namespace __impl { namespace hal { namespace opencl { 

gmacError_t compile_code(platform &plat, const std::string &code, const std::string &flags)
{
    /* module_descriptor *descriptor = */new module_descriptor(code, flags);

    gmacError_t ret;
    code_repository repository = module_descriptor::create_modules(plat, ret);

    Modules_.insert(map_platform_repository::value_type(&plat, repository));

    return ret;
}

gmacError_t compile_binary(platform &plat, const std::string &code, const std::string &flags)
{
    return gmacErrorFeatureNotSupported;
}

gmacError_t error(cl_int err)
{
    gmacError_t error = gmacSuccess;
    switch(err) {
        __GMAC_ERROR(CL_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CL_DEVICE_NOT_FOUND, gmacErrorNoAccelerator);
        __GMAC_ERROR(CL_DEVICE_NOT_AVAILABLE, gmacErrorInvalidAccelerator);
        __GMAC_ERROR(CL_MEM_OBJECT_ALLOCATION_FAILURE, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_HOST_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CL_OUT_OF_RESOURCES, gmacErrorMemoryAllocation);

        default: error = gmacErrorUnknown;
    }
    return error;
}

gmacError_t
stream_t::sync()
{
    TRACE(LOCAL, "stream <"FMT_ID">: waiting for stream", this->get_print_id());
    cl_int ret = clFinish((*this)());

    return error(ret);
}

void
_event_t::reset(bool async, type t)
{
    // Locking is not needed
    async_ = async;
    type_ = t;
    err_ = gmacSuccess;
    synced_ = false;
    state_ = None;
    cl_int res = clReleaseEvent(event_);
    ASSERTION(res == CL_SUCCESS);
    
    remove_triggers();
}

_event_t::state
_event_t::get_state()
{
    lock();

    if (state_ != End) {
        cl_int status;
        cl_int res = clGetEventInfo(event_,
                                    CL_EVENT_COMMAND_EXECUTION_STATUS,
                                    sizeof(cl_int),
                                    &status, NULL);
        if (res == CL_SUCCESS) {
            if (status == CL_QUEUED) {
                state_ = Queued;
            } else if (status == CL_SUBMITTED) {
                state_ = Submit;
            } else if (status == CL_RUNNING) {
                state_ = Start;
            } else if (status == CL_COMPLETE) {
                state_ = End;
            } else {
                FATAL("Unhandled value");
            }
        }
    }

    unlock();

    return state_;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
