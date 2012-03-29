#include <sys/sysinfo.h>

#include "config/common.h"

#include "phys/aspace.h"
#include "phys/processing_unit.h"
#include "phys/platform.h"
#include "module.h"

#define __GMAC_ERROR(r, err) case r: error = err; if(r == CUDA_ERROR_INVALID_HANDLE) abort(); break

namespace __impl { namespace hal {

static bool
hal_initialized = false;

static
cuda::phys::list_platform platforms;

gmacError_t
init()
{
    gmacError_t ret = gmacSuccess;

    if (hal_initialized == false) {
        TRACE(GLOBAL, "Initializing CUDA Driver API");
        if (cuInit(0) != CUDA_SUCCESS) {
            FATAL("Unable to init CUDA");
        }
        hal_initialized = true;
    } else {
        FATAL("Double HAL platform initialization");
    }

    return ret;
}

gmacError_t
fini()
{
    if (hal_initialized == false) {
        FATAL("HAL not initialized");
    }

    gmacError_t ret = gmacSuccess;

    hal_initialized = false;

    for (cuda::phys::platform *p : platforms) {
        delete p;
    }

    platforms.clear();

    return ret;
}

namespace phys {

cuda::phys::list_platform
get_platforms()
{
    if (hal_initialized == false) {
        FATAL("HAL not initialized");
    }

    if (platforms.size() == 0) {
        /////////////////////
        // Backend devices //
        /////////////////////
        cuda::phys::platform *p = new cuda::phys::platform();

        typedef std::set<CUdevice> set_device;
        typedef std::map<CUdevice, set_device> map_peer;
        map_peer peers;

        typedef std::map<CUdevice, cuda::phys::processing_unit> map_processing_unit;
        map_processing_unit pUnits;

        typedef std::set<set_device> set_peers;
        set_peers peerGroups;

        typedef std::map<CUdevice, cuda::phys::aspace *> map_aspace;
        map_aspace aspaces;
        typedef std::map<CUdevice, cuda::phys::memory *> map_memory;
        map_memory memories;

        int devCount = 0;
        int devRealCount = 0;

        CUresult err = cuDeviceGetCount(&devCount);
        if (err != CUDA_SUCCESS)
            FATAL("Error getting CUDA-enabled devices");

        TRACE(GLOBAL, "Found %d CUDA capable devices", devCount);

        // Dectect and register peer devices
        for (int i = 0; i < devCount; i++) {
            CUdevice cuDev;
            if (cuDeviceGet(&cuDev, i) != CUDA_SUCCESS)
                FATAL("Unable to access CUDA device");

            // Create compute device
            int attr = 0;
            if (cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev) != CUDA_SUCCESS)
                FATAL("Unable to access CUDA device");
            // Skip prohibited devices
            if (attr == CU_COMPUTEMODE_PROHIBITED) continue;

            set_device peerDevices;

            for (int j = 0; j < devCount; j++) {
                if (i != j) {
                    CUdevice cuDev2;
                    if (cuDeviceGet(&cuDev2, j) != CUDA_SUCCESS)
                        FATAL("Unable to access CUDA device");

                    // Create compute device
                    attr = 0;
                    if (cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev2) != CUDA_SUCCESS)
                        FATAL("Unable to access CUDA device");
                    // Skip prohibited devices
                    if (attr == CU_COMPUTEMODE_PROHIBITED) continue;

                    int canAccess;
                    err = cuDeviceCanAccessPeer(&canAccess, cuDev, cuDev2);
                    if (err != CUDA_SUCCESS)
                        FATAL("Error getting CUDA-enabled devices");

                    if (canAccess == 1) {
                        peerDevices.insert(cuDev2);
                    }
                }
            }

            peers.insert(map_peer::value_type(cuDev, peerDevices));
            devRealCount++;
        }

        // Dectect and register memories
        for (auto device : peers) {
            size_t bytes;
            if (cuDeviceTotalMem(&bytes, device.first) != CUDA_SUCCESS)
                FATAL("Unable to query CUDA device");

            // TODO: create a context to retrieve information about the memory
            cuda::phys::memory *mem = new cuda::phys::memory(*p, bytes);

            memories.insert(map_memory::value_type(device.first, mem));
        }

        // Dectect physical address spaces and memory connections for devices
        for (auto device : peers) {
            cuda::phys::aspace *pas;

            // Device with no peers
            if (device.second.size() == 0) {
                cuda::phys::aspace::set_memory deviceMemories;
                cuda::phys::memory *mem;

                ASSERTION(memories.count(device.first) == 1);
                mem = memories[device.first];

                cuda::phys::processing_unit::set_memory_connection connections;
                cuda::phys::processing_unit::memory_connection connection(*mem, true, 0);

                deviceMemories.insert(mem);
                connections.insert(connection);

                pas = new cuda::phys::aspace(*p, deviceMemories);

                cuda::phys::processing_unit *pUnit = new cuda::phys::processing_unit(*p, *pas, device.first);

                p->add_paspace(*pas);
                p->add_processing_unit(*pUnit);

                // Register the aspace
                aspaces.insert(map_aspace::value_type(device.first, pas));
            } else {
                // Not registered yet
                cuda::phys::aspace::set_memory deviceMemories;
                cuda::phys::memory *mem;

                ASSERTION(memories.count(device.first) == 1);
                mem = memories[device.first];

                cuda::phys::processing_unit::set_memory_connection connections;
                cuda::phys::processing_unit::memory_connection connection(*mem, true, 0);

                deviceMemories.insert(mem);
                connections.insert(connection);

                for (CUdevice dev : device.second) {
                    ASSERTION(memories.count(dev) == 1);

                    deviceMemories.insert(memories[dev]);

                    cuda::phys::processing_unit::memory_connection connection(*mem, true, 0);
                    connections.insert(connection);
                }

                map_aspace::iterator it = aspaces.find(device.first);
                if (it == aspaces.end()) {
                    pas = new cuda::phys::aspace(*p, deviceMemories);

                    // Register the aspace
                    aspaces.insert(map_aspace::value_type(device.first, pas));

                    // Register the aspace for the peers, too
                    for (auto peer : device.second) {
                        aspaces.insert(map_aspace::value_type(peer, pas));
                    }

                    p->add_paspace(*pas); 
                } else {
                    pas = it->second;
                }

                cuda::phys::processing_unit *pUnit = new cuda::phys::processing_unit(*p, *pas, device.first);

                p->add_processing_unit(*pUnit);
            }
        }

        if (devRealCount == 0)
            MESSAGE("No CUDA-enabled devices found");

        /////////////////////
        // Backend devices //
        /////////////////////
#if 0
        {
            for (int i = 0; i < get_nprocs(); ++i) {
                p->add_device(*new cuda::cpu(*p, *cpuDomain));
            }
        }
#endif

        platforms.push_back(p);
    }

    return platforms;
}

}

}}

namespace __impl { namespace hal { namespace cuda { 

gmacError_t error(CUresult err)
{
    gmacError_t error = gmacSuccess;
    switch(err) {
        __GMAC_ERROR(CUDA_SUCCESS, gmacSuccess);
        __GMAC_ERROR(CUDA_ERROR_INVALID_VALUE, gmacErrorInvalidValue);
        __GMAC_ERROR(CUDA_ERROR_OUT_OF_MEMORY, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NOT_INITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_DEINITIALIZED, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_NO_DEVICE, gmacErrorNoDevice);
        __GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, gmacErrorInvalidDevice);
        __GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, gmacErrorInvalidDeviceFunction);
        __GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, gmacErrorMemoryAllocation);
        __GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, gmacErrorInvalidDeviceFunction);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_ACQUIRED, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_FILE_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_INVALID_HANDLE, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_FOUND, gmacErrorApiFailureBase);
        __GMAC_ERROR(CUDA_ERROR_NOT_READY, gmacErrorNotReady);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_FAILED, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, gmacErrorLaunchFailure);
        __GMAC_ERROR(CUDA_ERROR_UNKNOWN, gmacErrorUnknown);
        default: error = gmacErrorUnknown;
    }
    return error;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
