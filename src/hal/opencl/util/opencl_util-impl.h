#ifndef GMAC_HAL_OPENCL_OPENCLUTIL_IMPL_H_
#define GMAC_HAL_OPENCL_OPENCLUTIL_IMPL_H_

#include <algorithm>

namespace __impl { namespace hal { namespace opencl { namespace util {

static std::string OpenCLNVidiaDevicePrefix[] = {
    "ION",
    "Tesla",
    "GeForce"
};

static std::string OpenCLAMDFusionDevice[] = {
    "Wrestler",
    "WinterPark",
    "BeaverCreek"
};

std::string
getPlatformString(int string, cl_platform_id id);

inline std::string
get_platform_name(cl_platform_id id)
{
    return getPlatformString(CL_PLATFORM_NAME, id);
}

inline std::string
get_platform_vendor(cl_platform_id id)
{
    return getPlatformString(CL_PLATFORM_VENDOR, id);
}

std::string
get_device_string(int string, cl_device_id id);

inline std::string
get_device_name(cl_device_id id)
{
    return get_device_string(CL_DEVICE_NAME, id);
}

inline std::string
get_device_vendor(cl_device_id id)
{
    return get_device_string(CL_DEVICE_VENDOR, id);
}

inline bool
is_device_amd_fusion(cl_device_id id)
{
    std::string *end = OpenCLAMDFusionDevice + 3;
    std::string *str = find(OpenCLAMDFusionDevice, end, get_device_name(id));
    return str != end;
}

}}}}

#endif
