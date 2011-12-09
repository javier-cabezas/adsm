#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <sstream>

#include <gmac/opencl.h>

typedef std::map<ecl_device_type, std::string> DevStrings;
static DevStrings devTypeString;

static ecl_device_type accTypes [] = { GMAC_DEVICE_TYPE_UNKNOWN,
                                       GMAC_DEVICE_TYPE_CPU,
                                       GMAC_DEVICE_TYPE_GPU,
                                       GMAC_DEVICE_TYPE_ACCELERATOR };

static void init_map()
{
    devTypeString.insert(DevStrings::value_type(GMAC_DEVICE_TYPE_UNKNOWN,     "GMAC_DEVICE_TYPE_UNKNOWN"));
    devTypeString.insert(DevStrings::value_type(GMAC_DEVICE_TYPE_CPU,         "GMAC_DEVICE_TYPE_CPU"));
    devTypeString.insert(DevStrings::value_type(GMAC_DEVICE_TYPE_GPU,         "GMAC_DEVICE_TYPE_GPU"));
    devTypeString.insert(DevStrings::value_type(GMAC_DEVICE_TYPE_ACCELERATOR, "GMAC_DEVICE_TYPE_ACCELERATOR"));
}

static std::string get_type_string(ecl_device_type type)
{
    std::string type_string;
    for (unsigned i = 0; i < 4; i++) {
        ecl_device_type t = accTypes[i];
        if ((type & t) != 0) {
            if (type_string.size() > 0) {
                type_string += " | ";
            }
            type_string.append(devTypeString[t]);
        }
    }

    return type_string;
}

static std::string get_dim_sizes_string(unsigned dims, const size_t *maxSizes)
{
    std::string dim_sizes_string;
    for (unsigned d = 0; d < dims; d++) {
        std::stringstream ss;
        ss << maxSizes[d];
        if (dim_sizes_string.size() > 0) {
            dim_sizes_string.append(", ");
        }
        dim_sizes_string.append(ss.str());
    }
    return dim_sizes_string;
}

int main(int argc, char *argv[])
{
    ecl_device_info info;

    init_map();

    for (unsigned i = 0; i < eclGetNumberOfDevices(); i++) {
        assert(eclGetDeviceInfo(i, &info) == eclSuccess);
        fprintf(stdout, "Accelerator %u/%u\n", i + 1, eclGetNumberOfDevices());

        fprintf(stdout, "- name: %s\n", info.deviceName);
        fprintf(stdout, "- vendor: %s\n", info.vendorName);
        fprintf(stdout, "- vendor id: %u\n", info.vendorId);
        fprintf(stdout, "- type: %s\n", get_type_string(info.deviceType).c_str());
        fprintf(stdout, "- available: %u\n", info.isAvailable);

        fprintf(stdout, "- compute units: %u\n", info.computeUnits);
        fprintf(stdout, "- max dimensions: %u\n", info.maxDimensions);
        fprintf(stdout, "- max sizes: %s\n", get_dim_sizes_string(info.maxDimensions, info.maxSizes).c_str());
        fprintf(stdout, "- max work group size: "FMT_SIZE"\n", info.maxWorkGroupSize);

        fprintf(stdout, "- global mem size: "FMT_SIZE"\n", info.globalMemSize);
        fprintf(stdout, "- local mem size: "FMT_SIZE"\n", info.localMemSize);
        fprintf(stdout, "- cache mem size: "FMT_SIZE" ("FMT_SIZE" per compute unit)\n", info.cacheMemSize, info.cacheMemSize / info.computeUnits);

        fprintf(stdout, "- driver version: %u.%u.%u\n", info.driverMajor, info.driverMinor, info.driverRev);
        fprintf(stdout, "\n");
    }

    return 0;
}
