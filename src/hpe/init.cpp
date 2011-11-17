/**
 * \file init.cpp
 *
 * Initialization routines
 */

#include "core/hpe/address_space.h"
#include "core/hpe/vdevice.h"
#include "core/hpe/process.h"
#include "core/hpe/thread.h"

#include "hal/types.h"

#include "memory/Allocator.h"
#include "memory/Handler.h"
#include "memory/manager.h"
#include "memory/Memory.h"
#include "memory/allocator/Slab.h"

#include "util/Parameter.h"
#include "util/Logger.h"

#include "trace/Tracer.h"

#include "init.h"

static gmac::core::hpe::process *Process_ = NULL;
static gmac::memory::manager *Manager_ = NULL;
static __impl::memory::Allocator *Allocator_ = NULL;

#if 0
extern void CUDA(gmac::core::hpe::Process &);
extern void OpenCL(gmac::core::hpe::Process &);
#endif

void initGmac(void)
{
    /* Call initialization of interpose libraries */
#if defined(POSIX)
    osInit();
    threadInit();
#endif
    stdcInit();

#ifdef USE_MPI
    mpiInit();
#endif

    TRACE(GLOBAL, "Using %s memory manager", config::params::Protocol);
    TRACE(GLOBAL, "Using %s memory allocator", config::params::Allocator);

    // Set the entry and exit points for Manager
    __impl::memory::Handler::setEntry(enterGmac);
    __impl::memory::Handler::setExit(exitGmac);

    isRunTimeThread_.set(&privateFalse);
    // Process is a singleton class. The only allowed instance is Proc_
    TRACE(GLOBAL, "Initializing process");
    Process_ = new gmac::core::hpe::process();

    TRACE(GLOBAL, "Initializing memory");
    Manager_ = new gmac::memory::manager(*Process_);
#if !defined(USE_OPENCL)
    Allocator_ = new __impl::memory::allocator::Slab(*Manager_);
#endif

#if 0
#if defined(USE_CUDA)
    TRACE(GLOBAL, "Initializing CUDA");
    CUDA(*Process_);
#endif
#if defined(USE_OPENCL)
    TRACE(GLOBAL, "Initializing OpenCL");
    OpenCL(*Process_);
#endif
#endif
    gmacError_t err = __impl::hal::init_platform();
    CFATAL(err == gmacSuccess, "Error initializing the platform");

    std::list<__impl::hal::device *> devices = __impl::hal::init_devices();

    std::list<__impl::hal::device *>::iterator it;

    for (it = devices.begin(); it != devices.end(); it++) {
        Process_->get_resource_manager().register_device(**it);
    }

    Process_->init();
}

namespace __impl {
    namespace core {
        core::process &getProcess() { return *Process_; }
        namespace hpe {
            //Mode &getCurrentVirtualDevice() { return __impl::core::hpe::thread::get_current_thread().getCurrentVirtualDevice(); }
            process &getProcess() { return *Process_; }
        }
        vdevice &get_virtual_device() { return __impl::core::hpe::thread::get_current_thread().get_current_virtual_device(); }
    }

    namespace memory {
        manager &getManager() { return *Manager_; }
        bool hasAllocator() { return Allocator_ != NULL; }
        Allocator &getAllocator() { return *Allocator_; }
    }
}


// We cannot call the destructor because the backend might have
// been uninitialized
#if 0
DESTRUCTOR(fini);
static void fini(void)
{
    enterGmac();
    if(AtomicInc(gmacFini__) == 0) {
        Allocator_->destroy();
        Manager_->destroy();
        Process_->destroy();
        delete inGmacLock;
    }
    // TODO: Clean-up logger
}
#endif


#if defined(_WIN32)
#include <windows.h>

static void InitThread(const bool &isRunTimeThread)
{
    gmac::trace::StartThread("CPU");
    isRunTimeThread_.set(&isRunTimeThread);
    enterGmac();
    __impl::core::hpe::getProcess().initThread();
    gmac::trace::SetThreadState(__impl::trace::Running);
    exitGmac();
}

static void FiniThread()
{
    enterGmac();
    gmac::trace::SetThreadState(gmac::trace::Idle);
    // Modes and Contexts already destroyed in Process destructor
    __impl::core::hpe::getProcess().finiThread();
    exitGmac();
}

#include <Winternl.h>
#include <Psapi.h>

// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
    typedef NTSTATUS (WINAPI *myfun)(HANDLE h, THREADINFOCLASS t, PVOID p, ULONG u, PULONG l);
    static myfun NtQueryInformationThread_ = NULL;
    static HMODULE dll = NULL;

    switch(dwReason) {
    case DLL_PROCESS_ATTACH:
        break;
    case DLL_PROCESS_DETACH:
        {
            if (NtQueryInformationThread_ != NULL) {
                BOOL ret = FreeLibrary(dll);
                CFATAL(ret != FALSE, "Error unloading library");
            }
        }
        break;
    case DLL_THREAD_ATTACH:
        {
            static DWORD pid;
            static HANDLE processHandle;

            static void *openCLStartAddr;
            static void *openCLEndAddr;
            static void *nvCUDAStartAddr;
            static void *nvCUDAEndAddr;

            if (NtQueryInformationThread_ == NULL) {
                // Load ntdll library since we need NtQueryInformationThread
                dll = LoadLibrary("ntdll.dll");
                CFATAL(dll != NULL, "Error loading ntdll.dll");
                NtQueryInformationThread_ = (myfun) GetProcAddress(dll, "NtQueryInformationThread");
                CFATAL(NtQueryInformationThread_ != NULL, "Error finding NtQueryInformationThread");

                // Get process handler
                pid = GetCurrentProcessId();
                processHandle = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, false, pid);
                CFATAL(processHandle != NULL, "Error opening process");

                // Get OpenCL.dll and nvcuda.dll memory load addresses
                HMODULE libHandle = GetModuleHandle("OpenCL.dll");
                CFATAL(libHandle != NULL);
                MODULEINFO libInfo;
                BOOL ret = GetModuleInformation(processHandle, libHandle, &libInfo, sizeof(libInfo));
                CFATAL(ret != FALSE, "Error getting OpenCL.dll information");
                openCLStartAddr = libHandle;
                openCLEndAddr = (void *) (LPCCH(libHandle) + libInfo.SizeOfImage);

                libHandle = GetModuleHandle("nvcuda.dll");
                if (libHandle != NULL) {
                    ret = GetModuleInformation(processHandle, libHandle, &libInfo, sizeof(libInfo));
                    CFATAL(ret != FALSE, "Error getting nvcuda.dll information");
                    nvCUDAStartAddr = libHandle;
                    nvCUDAEndAddr = (void *) (LPCCH(libHandle) + libInfo.SizeOfImage);
                }

                // Free resources
                ret = CloseHandle(processHandle);
                CFATAL(ret != FALSE, "Error closing process handle");
            }

            // Get thread start address
            DWORD tid = GetCurrentThreadId();
            HANDLE threadHandle = OpenThread(THREAD_QUERY_INFORMATION, false, tid);
            CFATAL(threadHandle != NULL, "Error opening thread");
            void *threadStartAddr = NULL;

            ULONG sizeInfo;
            NTSTATUS status;
            status = NtQueryInformationThread_(threadHandle, THREADINFOCLASS(9), &threadStartAddr, sizeof(void *), &sizeInfo);
            CFATAL(status == 0);

            bool isRunTimeThread = (threadStartAddr >= openCLStartAddr && threadStartAddr < openCLEndAddr) ||
                                   (threadStartAddr >= nvCUDAStartAddr && threadStartAddr < nvCUDAEndAddr);

            BOOL ret = CloseHandle(threadHandle);
            CFATAL(ret != FALSE, "Error closing thread handle");

            if(isRunTimeThread) InitThread(privateTrue);
			else InitThread(privateFalse);
        }
        break;
    case DLL_THREAD_DETACH:
        FiniThread();
        break;
    };
    return TRUE;
}


#endif


/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
