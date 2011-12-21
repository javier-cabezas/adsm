#include "trace/logger.h"

#include "hal/types.h"
#include "device.h"

namespace __impl { namespace hal { namespace opencl {

platform::platform(cl_platform_id id, cl_context ctx) :
    openclPlatformId_(id),
    ctx_(ctx)
{
}

cl_platform_id
platform::get_cl_platform_id() const
{
    return openclPlatformId_;
}

void
platform::add_device(device &d)
{
    devices_.push_back(&d);
}

unsigned
platform::get_ndevices()
{
    return devices_.size();
}

cl_device_id *
platform::get_cl_device_array()
{
    cl_device_id *deviceIds = new cl_device_id[devices_.size()];

    unsigned i = 0;
    for (std::list<device *>::iterator it  = devices_.begin();
                                       it != devices_.end();
                                     ++it) {
        deviceIds[i++] = (*it)->openclDeviceId_;
    }
    return deviceIds;
}

cl_context
platform::get_context()
{
    return ctx_;
}

device::device(platform &p,
               cl_device_id openclDeviceId,
               coherence_domain &coherenceDomain) :
    Parent(coherenceDomain),
    gmac::util::mutex<device>("device"),
    platform_(p),
    openclDeviceId_(openclDeviceId),
    context_(p.get_context()),
    isInfoInitialized_(false)
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
        if (integrated_) {
        	TRACE(GLOBAL, "Device uses host memory");
        } else {
        	TRACE(GLOBAL, "Device does NOT use host memory");
        }
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

gmacError_t
device::get_info(GmacDeviceInfo &info)
{
	lock();
	if (!isInfoInitialized_) {
		cl_device_type deviceType;
		cl_uint deviceVendor;
		char *deviceName, *vendorName;

		size_t nameSize;
		cl_int res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_NAME, 0, NULL, &nameSize);
		ASSERTION(res == CL_SUCCESS);
		deviceName = new char[nameSize + 1];
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_NAME, nameSize, deviceName, NULL);
		ASSERTION(res == CL_SUCCESS);
		deviceName[nameSize] = '\0';
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_VENDOR, 0, NULL, &nameSize);
		ASSERTION(res == CL_SUCCESS);
		vendorName = new char[nameSize + 1];
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_VENDOR, nameSize, vendorName, NULL);
		ASSERTION(res == CL_SUCCESS);
		vendorName[nameSize] = '\0';

		info_.deviceName = deviceName;
		info_.vendorName = vendorName;

		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_TYPE, sizeof(cl_device_type), &deviceType, NULL);
		ASSERTION(res == CL_SUCCESS);

		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), &deviceVendor, NULL);
		ASSERTION(res == CL_SUCCESS);

		info.vendorId = unsigned(deviceVendor);

		info_.deviceType = GmacDeviceType(0);
		if (deviceType & CL_DEVICE_TYPE_CPU) {
			info_.deviceType = GmacDeviceType(info_.deviceType | GMAC_DEVICE_TYPE_CPU);
		}

		if (deviceType & CL_DEVICE_TYPE_GPU) {
			info_.deviceType = GmacDeviceType(info_.deviceType | GMAC_DEVICE_TYPE_GPU);
		}

		if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
			info_.deviceType = GmacDeviceType(info_.deviceType | GMAC_DEVICE_TYPE_ACCELERATOR);
		}

		/// \todo Compute this value
		info_.isAvailable = 1;

        cl_uint computeUnits;
        cl_uint dimensions;
        size_t workGroupSize;
        cl_ulong globalMemSize;
        cl_ulong localMemSize;
        cl_ulong cacheMemSize;
        size_t *maxSizes;

		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &computeUnits, NULL);
		ASSERTION(res == CL_SUCCESS);
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &dimensions, NULL);
		ASSERTION(res == CL_SUCCESS);

		info_.computeUnits = computeUnits;
		info_.maxDimensions = dimensions;
		maxSizes = new size_t[dimensions];

		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * dimensions, maxSizes, NULL);
		ASSERTION(res == CL_SUCCESS);

		info_.maxSizes = maxSizes;

		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workGroupSize, NULL);
		ASSERTION(res == CL_SUCCESS);
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &globalMemSize, NULL);
		ASSERTION(res == CL_SUCCESS);
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemSize, NULL);
		ASSERTION(res == CL_SUCCESS);
		res = clGetDeviceInfo(openclDeviceId_, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), &cacheMemSize, NULL);
		ASSERTION(res == CL_SUCCESS);
		info_.maxWorkGroupSize = workGroupSize;
		info_.globalMemSize = static_cast<size_t>(globalMemSize);
		info_.localMemSize  = static_cast<size_t>(localMemSize);
		info_.cacheMemSize  = static_cast<size_t>(cacheMemSize);

        size_t driverSize = 0;
        res = clGetDeviceInfo(openclDeviceId_, CL_DRIVER_VERSION, 0, NULL, &driverSize);
        ASSERTION(res == CL_SUCCESS);
        if(driverSize > 0) {
            char *driverName = new char[driverSize + 1];
            res = clGetDeviceInfo(openclDeviceId_, CL_DRIVER_VERSION, driverSize, driverName, NULL);
            ASSERTION(res == CL_SUCCESS);
            std::string driverString(driverName);
            size_t number = driverString.find_first_of("1234567890");
            size_t first_dot = driverString.find_first_of('.');
            size_t last_dot = driverString.find_last_of('.');
            if(last_dot == first_dot) last_dot = driverString.length() + 1;
            if(first_dot != std::string::npos) {
                std::string majorString = driverString.substr(number, first_dot);
                info_.driverMajor = atoi(majorString.c_str());
                std::string minorString = driverString.substr(first_dot + 1, last_dot);
                info_.driverMinor = atoi(minorString.c_str());
                if(last_dot < driverString.length()) {
                    std::string revString = driverString.substr(last_dot + 1);
                    info_.driverRev = atoi(revString.c_str());
                }
                else info_.driverRev = 0;
            }
            delete []driverName;
        }

		isInfoInitialized_ = true;
	}
	unlock();

	info = info_;

	return gmacSuccess;
}

platform &
device::get_platform()
{
    return platform_;
}

const platform &
device::get_platform() const
{
    return platform_;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
