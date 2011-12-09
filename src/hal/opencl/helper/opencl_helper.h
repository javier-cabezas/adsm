#ifndef GMAC_HAL_OPENCL_OPENCLUTIL_H_
#define GMAC_HAL_OPENCL_OPENCLUTIL_H_

#include <string>
#include <vector>

#include "config/common.h"

namespace __impl { namespace hal { namespace opencl { namespace helper {

enum GMAC_LOCAL opencl_vendor {
    VENDOR_AMD,
    VENDOR_NVIDIA,
    VENDOR_INTEL,
    VENDOR_UNKNOWN
};

enum GMAC_LOCAL opencl_platform {
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
    PLATFORM_INTEL,
    PLATFORM_APPLE,
    PLATFORM_UNKNOWN
};

typedef std::pair<unsigned, unsigned> opencl_version;

// Platform functions
std::string GMAC_LOCAL
get_platform_name(cl_platform_id id);

std::string GMAC_LOCAL
get_platform_vendor(cl_platform_id id);

opencl_platform GMAC_LOCAL
get_platform(cl_platform_id id);

opencl_version GMAC_LOCAL
get_opencl_version(cl_platform_id id);

// Device functions
std::string GMAC_LOCAL
get_device_name(cl_device_id id);

std::string GMAC_LOCAL
get_device_vendor(cl_device_id id);

bool GMAC_LOCAL
is_device_amd_fusion(cl_device_id id);

// Context functions
cl_device_id GMAC_LOCAL
get_context_device(cl_context context);

std::vector<cl_device_id> GMAC_LOCAL
get_context_devices(cl_context context);

// Commmand queue functions
cl_device_id GMAC_LOCAL
get_queue_device(cl_command_queue queue);

cl_context GMAC_LOCAL
get_queue_context(cl_command_queue queue);

}}}}

#include "opencl_helper-impl.h"

#endif
