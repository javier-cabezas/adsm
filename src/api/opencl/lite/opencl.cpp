#if defined(POSIX)
#include "os/posix/loader.h"
#elif defined(WINDOWS)
#include "os/windows/loader.h"
#endif

#include "config/config.h"

#include "api/opencl/lite/Process.h"
#include "include/gmac/lite.h"
#include "memory/Allocator.h"
#include "memory/Handler.h"
#include "memory/Manager.h"
#include "util/Logger.h"
#include "util/Parameter.h"

#include <CL/cl.h>

#if defined(__GNUC__)
#define RETURN_ADDRESS __builtin_return_address(0)
#elif defined(_MSC_VER)
extern "C" void * _ReturnAddress(void);
#pragma intrinsic(_ReturnAddress)
#define RETURN_ADDRESS _ReturnAddress()
static long getpagesize (void) {
    static long pagesize = 0;
    if(pagesize == 0) {
        SYSTEM_INFO systemInfo;
        GetSystemInfo(&systemInfo);
        pagesize = systemInfo.dwPageSize;
    }
    return pagesize;
}
#endif


using __impl::memory::Handler;
using __impl::opencl::lite::Process;
using __impl::opencl::lite::Mode;
using __impl::util::Private;

static Private<const char> inGmac_;
static const char gmacCode = 1;
static const char userCode = 0;
static Atomic gmacInit_ = 0;



SYM(cl_context, __opencl_clCreateContext,
        const cl_context_properties *,
        cl_uint,
        const cl_device_id *,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

SYM(cl_context, __opencl_clCreateContextFromType,
        const cl_context_properties *,
        cl_device_type,
        void (CL_CALLBACK *)(const char *, const void *, size_t, void *),
        void *,
        cl_int *);

SYM(cl_int, __opencl_clRetainContext, cl_context);

SYM(cl_int, __opencl_clReleaseContext, cl_context);

SYM(cl_command_queue, __opencl_clCreateCommandQueue,
        cl_context,
        cl_device_id,
        cl_command_queue_properties,
        cl_int *);

SYM(cl_int, __opencl_clRetainCommandQueue, cl_command_queue);

SYM(cl_int, __opencl_clReleaseCommandQueue, cl_command_queue);

SYM(cl_int, __opencl_clEnqueueNDRangeKernel,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const size_t *,
        const size_t *,
        const size_t *,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clEnqueueTask,
        cl_command_queue,
        cl_kernel,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clEnqueueNativeKernel,
        cl_command_queue,
        void (*)(void *),
        void *,
        size_t,
        cl_uint,
        const cl_mem *,
        const void **,
        cl_uint,
        const cl_event *,
        cl_event *);

SYM(cl_int, __opencl_clFinish, cl_command_queue);

#ifdef __cplusplus
extern "C" {
#endif

CONSTRUCTOR(init);
void openclInit();

static void enterGmac()
{
    if(AtomicTestAndSet(gmacInit_, 0, 1) == 0) init();
    inGmac_.set(&gmacCode);
}

static void exitGmac()
{
    inGmac_.set(&userCode);
}

static bool inGmac()
{
    if(gmacInit_ == 0) return 1;
    char *ret = (char *)inGmac_.get();
    if(ret == NULL) return false;
    else if(*ret == gmacCode) return true;
    return false;
}


cl_context SYMBOL(clCreateContext)(
        const cl_context_properties *properties,
        cl_uint num_devices,
        const cl_device_id *devices,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContext(properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
    if(inGmac() || *errcode_ret != CL_SUCCESS) return ret;

    enterGmac();
    Process &proc = Process::getInstance<Process>();
    proc.createMode(ret, num_devices, devices);
    exitGmac();

    return ret;
}

cl_context SYMBOL(clCreateContextFromType)(
        const cl_context_properties *properties,
        cl_device_type device_type,
        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
        void *user_data,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateContext == NULL) openclInit();
    cl_context ret = __opencl_clCreateContextFromType(properties, device_type, pfn_notify, user_data, errcode_ret);
    if(inGmac() || *errcode_ret != CL_SUCCESS) return ret;
    return ret;
}
        
cl_int SYMBOL(clRetainContext)(cl_context context)
{
    if(__opencl_clRetainContext == NULL) openclInit();
    cl_int ret = __opencl_clRetainContext(context);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    Process::getInstance<Process>().getMode(context);
    exitGmac();
    return ret;
}

cl_int SYMBOL(clReleaseContext)(cl_context context)
{
    if(__opencl_clReleaseContext == NULL) openclInit();
    cl_int ret = __opencl_clReleaseContext(context);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode != NULL) {
        mode->release();
        // We release the mode twice to effectively decrease the usage count
        mode->release();
    }
    exitGmac();
    return ret;
}

cl_command_queue SYMBOL(clCreateCommandQueue)(
        cl_context context,
        cl_device_id device,
        cl_command_queue_properties properties,
        cl_int *errcode_ret)
{
    if(__opencl_clCreateCommandQueue == NULL) openclInit();
    cl_command_queue ret = __opencl_clCreateCommandQueue(context, device, properties,  errcode_ret);
    if(inGmac() || *errcode_ret != CL_SUCCESS) return ret;
    enterGmac();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode == NULL) return ret;
    mode->addQueue(ret);
    mode->release();
    exitGmac();
    return ret;
}

cl_int SYMBOL(clRetainCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clRetainCommandQueue == NULL) openclInit();
    cl_int ret = __opencl_clRetainCommandQueue(command_queue);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    return ret;
}

cl_int SYMBOL(clReleaseCommandQueue)(cl_command_queue command_queue)
{
    if(__opencl_clReleaseCommandQueue == NULL) openclInit();
    cl_context context;
    cl_int ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret != CL_SUCCESS) return ret;

    cl_uint count = 0;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_REFERENCE_COUNT, sizeof(cl_uint), &count, NULL);
    if(ret != CL_SUCCESS) return ret;

    ret = __opencl_clRetainCommandQueue(command_queue);
    if(inGmac() || ret != CL_SUCCESS || count > 1) return ret;
    enterGmac();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode == NULL) return ret;
    mode->removeQueue(command_queue);
    mode->release();
    exitGmac();
    return ret;
}

#if 0
static void acquireMemoryObjects(cl_event event, cl_int status, void *user_data)
{
    Mode *mode = NULL;
    cl_context context;
    cl_command_queue queue;
    cl_int ret = CL_SUCCESS;
    ret = clGetEventInfo(event, CL_EVENT_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret != CL_SUCCESS) goto do_exit;
    ret = clGetEventInfo(event, CL_EVENT_COMMAND_QUEUE, sizeof(cl_command_queue), &queue, NULL);
    if(ret != CL_SUCCESS) goto do_exit;

    mode = Process::getInstance<Process>().getMode(context);
    if(mode != NULL) {
        mode->setActiveQueue(queue);
        gmac::memory::Manager::getInstance().acquireObjects(*mode);
        mode->deactivateQueue();
        mode->release();
    }

do_exit:
    cl_event *user_event = (cl_event *)user_data;
    if(user_event != NULL) delete user_event;
}
#endif

static cl_int releaseMemoryObjects(cl_command_queue command_queue)
{
    cl_int ret = CL_SUCCESS;
    cl_context context;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);
    if(ret == CL_SUCCESS) {
        Mode *mode = Process::getInstance<Process>().getMode(context);
        if(mode != NULL) {
            mode->setActiveQueue(command_queue);
            gmac::memory::Manager::getInstance().releaseObjects(*mode);
            mode->deactivateQueue();
            mode->release();
        }
    }
    return ret;
}


cl_int SYMBOL(clEnqueueNDRangeKernel)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint work_dim,
    const size_t *global_work_offset,
    const size_t *global_work_size,
    const size_t *local_work_size,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueNDRangeKernel == NULL) openclInit();
    enterGmac();
    cl_int ret = releaseMemoryObjects(command_queue);
    /* cl_event *user_event = NULL; */
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueNDRangeKernel(command_queue, kernel, work_dim, global_work_offset,
        global_work_size, local_work_size, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int SYMBOL(clEnqueueTask)(
    cl_command_queue command_queue,
    cl_kernel kernel,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{ 
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueTask == NULL) openclInit();
    enterGmac();
    /* cl_event *user_event = NULL; */
    cl_int ret = releaseMemoryObjects(command_queue);
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueTask(command_queue, kernel, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int SYMBOL(clEnqueueNativeKernel)(
    cl_command_queue command_queue,
    void (*user_func)(void *),
    void *args,
    size_t cb_args,
    cl_uint num_mem_objects,
    const cl_mem *mem_list,
    const void **args_mem_loc,
    cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list,
    cl_event *event)
{
    ASSERTION(inGmac() == false);
    if(__opencl_clEnqueueNativeKernel == NULL) openclInit();
    /* cl_event *user_event = NULL; */
    cl_int ret = releaseMemoryObjects(command_queue);
    if(ret != CL_SUCCESS) goto do_exit;
    /*
    if(event == NULL) user_event = new cl_event();
    else user_event = event;
    */
    ret = __opencl_clEnqueueNativeKernel(command_queue, user_func, args, cb_args, num_mem_objects,
        mem_list, args_mem_loc, num_events_in_wait_list, event_wait_list, event);
    if(ret != CL_SUCCESS) goto do_exit;
    /* ret = clSetEventCallback(*user_event, CL_COMPLETE, acquireMemoryObjects, event); */

do_exit:
    exitGmac();
    return ret;
}

cl_int SYMBOL(clFinish)(cl_command_queue command_queue)
{
    if(__opencl_clFinish == NULL) openclInit();
    cl_int ret = __opencl_clFinish(command_queue);
    if(inGmac() || ret != CL_SUCCESS) return ret;
    enterGmac();
    cl_context context;
    ret = clGetCommandQueueInfo(command_queue, CL_QUEUE_CONTEXT, sizeof(cl_context), &context, NULL);

    if(ret == CL_SUCCESS) {
        Mode *mode = Process::getInstance<Process>().getMode(context);
        if(mode != NULL) {
            mode->setActiveQueue(command_queue);
            gmac::memory::Manager::getInstance().acquireObjects(*mode);
            mode->deactivateQueue();
            mode->release();
        }
    }
    exitGmac();
    return ret;
}


GMAC_API cl_int clMalloc(cl_context context, void **addr, size_t count)
{
    cl_int ret = CL_SUCCESS;
    *addr = NULL;
    if(count == 0) return ret;

    enterGmac();
    gmac::trace::EnterCurrentFunction();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode != NULL) {
        __impl::memory::Allocator &allocator = __impl::memory::Allocator::getInstance();
        if(count < (__impl::util::params::ParamBlockSize / 2)) {
            *addr = allocator.alloc(*mode, count, hostptr_t(RETURN_ADDRESS));
        }
        else {
        	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
    	    count = (int(count) < getpagesize())? getpagesize(): count;
            ret = manager.alloc(*mode, (hostptr_t *) addr, count);
            mode->release();
        }
    }
    else ret = CL_INVALID_CONTEXT;
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;
}

GMAC_API cl_int clFree(cl_context context, void *addr)
{
    cl_int ret = CL_SUCCESS;
	enterGmac();
    gmac::trace::EnterCurrentFunction();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode != NULL) {
        __impl::memory::Allocator &allocator = __impl::memory::Allocator::getInstance();
        if(allocator.free(*mode, hostptr_t(addr)) == false) {
        	gmac::memory::Manager &manager = gmac::memory::Manager::getInstance();
            ret = manager.free(*mode, hostptr_t(addr));
        }
        mode->release();
    }
    else ret = CL_INVALID_CONTEXT;
    
    gmac::trace::ExitCurrentFunction();
	exitGmac();
	return ret;

}

GMAC_API cl_mem clBuffer(cl_context context, const void *ptr)
{
    accptr_t ret = accptr_t(0);
    enterGmac();
    Mode *mode = Process::getInstance<Process>().getMode(context);
    if(mode != NULL) ret = __impl::memory::Manager::getInstance().translate(*mode, hostptr_t(ptr));
    exitGmac();
    return ret.get();
}

static void openclInit()
{
    LOAD_SYM(__opencl_clCreateContext, clCreateContext);
    LOAD_SYM(__opencl_clCreateContextFromType, clCreateContextFromType);
    LOAD_SYM(__opencl_clRetainContext, clRetainContext);
    LOAD_SYM(__opencl_clReleaseContext, clReleaseContext);

    LOAD_SYM(__opencl_clCreateCommandQueue, clCreateCommandQueue);
    LOAD_SYM(__opencl_clRetainCommandQueue, clRetainCommandQueue);
    LOAD_SYM(__opencl_clReleaseCommandQueue, clReleaseCommandQueue);

    LOAD_SYM(__opencl_clEnqueueNDRangeKernel, clEnqueueNDRangeKernel);
    LOAD_SYM(__opencl_clEnqueueTask, clEnqueueTask);
    LOAD_SYM(__opencl_clEnqueueNativeKernel, clEnqueueNativeKernel);

    LOAD_SYM(__opencl_clFinish, clFinish);
}


static void init()
{
    Private<const char>::init(inGmac_);
    AtomicInc(gmacInit_);
    enterGmac();

    TRACE(GLOBAL, "Initializing Memory Manager");
    Handler::setEntry(enterGmac);
    Handler::setExit(exitGmac);

    TRACE(GLOBAL, "Initializing Process");
    Process::create<Process>();
    exitGmac();
}

#if defined(_WIN32)
#include <windows.h>


// DLL entry function (called on load, unload, ...)
BOOL APIENTRY DllMain(HANDLE /*hModule*/, DWORD dwReason, LPVOID /*lpReserved*/)
{
	switch(dwReason) {
		case DLL_PROCESS_ATTACH:
			openclInit();
            break;
		case DLL_PROCESS_DETACH:            
			break;
		case DLL_THREAD_ATTACH:
			break;
		case DLL_THREAD_DETACH:			
			break;
	};
    return TRUE;
}

#endif

#ifdef __cplusplus
}
#endif
