#include <sys/sysinfo.h>

#include "config/common.h"

#include "phys/aspace.h"
#include "phys/processing_unit.h"
#include "phys/platform.h"
#include "code/module.h"

#define __GMAC_ERROR(r, err) case r: ret = err; break

namespace __impl { namespace hal {

static bool
hal_initialized = false;

static
cuda::phys::list_platform platforms;

hal::error
init()
{
    hal::error ret = hal::error::HAL_SUCCESS;

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

hal::error
fini()
{
    if (hal_initialized == false) {
        FATAL("HAL not initialized");
    }

    hal::error ret = hal::error::HAL_SUCCESS;

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
        detail::phys::memory *memoryHost;

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
            detail::phys::memory *mem = new detail::phys::memory(*p, bytes);

            memories.insert(map_memory::value_type(device.first, mem));
        }

        // Add host memory
        // TODO: detect host memory size
        memoryHost = new detail::phys::memory(*p, size_t(16) * 1024 * 1024 * 1024);

        // Dectect physical address spaces and memory connections for devices
        for (auto device : peers) {
            cuda::phys::aspace *pas;

            // Device with no peers
            if (device.second.size() == 0) {
                MESSAGE("GPU device 0 peers");
                detail::phys::aspace::set_memory deviceMemories;
                detail::phys::memory *mem;

                ASSERTION(memories.count(device.first) == 1);
                mem = memories[device.first];

                detail::phys::processing_unit::set_memory_connection connections;
                detail::phys::processing_unit::memory_connection connection(*mem, true, 0);

                // TODO: Detect integrated devices/memories
                // TODO: Compute access latency
                detail::phys::processing_unit::memory_connection connectionHost(*memoryHost, false, 500);

                deviceMemories.insert(mem);
                deviceMemories.insert(memoryHost);

                connections.insert(connection);
                connections.insert(connectionHost);

                // Create physical address space
                pas = new cuda::phys::aspace(*p, deviceMemories);

                cuda::phys::processing_unit *pUnit = new cuda::phys::processing_unit(*p, *pas, device.first);

                p->add_paspace(*pas);
                p->add_processing_unit(*pUnit);

                // Register memory connections for the device
                for (auto &conn : connections) {
                    pUnit->add_memory_connection(conn);
                }

                // Register the aspace
                aspaces.insert(map_aspace::value_type(device.first, pas));
            } else {
                MESSAGE("GPU device " FMT_SIZE " peers", device.second.size());
                // Not registered yet
                cuda::phys::aspace::set_memory deviceMemories;
                cuda::phys::memory *mem;

                ASSERTION(memories.count(device.first) == 1);
                mem = memories[device.first];

                detail::phys::processing_unit::set_memory_connection connections;
                detail::phys::processing_unit::memory_connection connection(*mem, true, 0);

                // TODO: Detect integrated devices/memories
                // TODO: Compute access latency
                detail::phys::processing_unit::memory_connection connectionHost(*memoryHost, false, 500);

                deviceMemories.insert(mem);
                deviceMemories.insert(memoryHost);

                connections.insert(connection);
                connections.insert(connectionHost);

                for (CUdevice dev : device.second) {
                    ASSERTION(memories.count(dev) == 1);

                    deviceMemories.insert(memories[dev]);

                    cuda::phys::processing_unit::memory_connection connection(*mem, true, 0);
                    connections.insert(connection);
                }

                map_aspace::iterator it = aspaces.find(device.first);
                if (it == aspaces.end()) {
                    // Create new physical address space
                    pas = new cuda::phys::aspace(*p, deviceMemories);

                    // Register the aspace
                    aspaces.insert(map_aspace::value_type(device.first, pas));

                    // Register the aspace for the peers, too
                    for (auto peer : device.second) {
                        aspaces.insert(map_aspace::value_type(peer, pas));
                    }

                    p->add_paspace(*pas); 
                } else {
                    // Someone already created physical address space
                    pas = it->second;
                }

                cuda::phys::processing_unit *pUnit = new cuda::phys::processing_unit(*p, *pas, device.first);

                p->add_processing_unit(*pUnit);

                // Register memory connections for the device
                for (auto &conn : connections) {
                    pUnit->add_memory_connection(conn);
                }
            }
        }

        // Create host physical address space
        detail::phys::aspace::set_memory hostMemories;
        hostMemories.insert(memoryHost);
        cpu::phys::aspace *pas = new cpu::phys::aspace(*p, hostMemories);
        // Create host CPUs
        {
            for (int i = 0; i < get_nprocs(); ++i) {
                //p->add_device(*new cuda::cpu(*p, *cpuDomain));
                cpu::phys::processing_unit *pUnit = new cpu::phys::processing_unit(*p, *pas);
                p->add_processing_unit(*pUnit);

                cpu::phys::processing_unit::memory_connection connection(*memoryHost, true, 0);
                pUnit->add_memory_connection(connection);
            }
        }

        if (devRealCount == 0)
            MESSAGE("No CUDA-enabled devices found");

        platforms.push_back(p);
    }

    return platforms;
}

}

}}

namespace __impl { namespace hal { namespace cuda { 

hal::error error_to_hal(CUresult err)
{
    hal::error ret = hal::error::HAL_SUCCESS;
    switch(err) {
        __GMAC_ERROR(CUDA_SUCCESS, hal::error::HAL_SUCCESS);
        __GMAC_ERROR(CUDA_ERROR_INVALID_VALUE, hal::error::HAL_ERROR_INVALID_VALUE);
        __GMAC_ERROR(CUDA_ERROR_OUT_OF_MEMORY, hal::error::HAL_ERROR_OUT_OF_RESOURCES);
        __GMAC_ERROR(CUDA_ERROR_NOT_INITIALIZED, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_DEINITIALIZED, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_NO_DEVICE, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_INVALID_DEVICE, hal::error::HAL_ERROR_INVALID_DEVICE);
        __GMAC_ERROR(CUDA_ERROR_INVALID_IMAGE, hal::error::HAL_ERROR_INVALID_FUNCTION);
        __GMAC_ERROR(CUDA_ERROR_INVALID_CONTEXT, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_INVALID_HANDLE, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_CONTEXT_ALREADY_CURRENT, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_MAPPED, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_NO_BINARY_FOR_GPU, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_ALREADY_ACQUIRED, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_FILE_NOT_FOUND, hal::error::HAL_ERROR_FILE_NOT_FOUND);
        __GMAC_ERROR(CUDA_ERROR_NOT_FOUND, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_NOT_READY, hal::error::HAL_ERROR_BACKEND);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, hal::error::HAL_ERROR_OUT_OF_RESOURCES);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_FAILED, hal::error::HAL_ERROR_FUNCTION);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_TIMEOUT, hal::error::HAL_ERROR_FUNCTION);
        __GMAC_ERROR(CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING, hal::error::HAL_ERROR_FUNCTION);
        __GMAC_ERROR(CUDA_ERROR_UNKNOWN, hal::error::HAL_ERROR_BACKEND);
        default: ret = hal::error::HAL_ERROR_BACKEND;
    }
    return ret;
}

}}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
