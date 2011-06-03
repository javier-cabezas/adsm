#include "api/opencl/hpe/Accelerator.h"
#include "api/opencl/hpe/Mode.h"

#include "api/opencl/IOBuffer.h"
#include "api/opencl/Tracer.h"

#include "core/Process.h"

#include "hpe/init.h"

#ifndef _MSC_VER
// Symbols needed for automatic compilation of embedded code
extern "C" {
    extern char __ocl_code_start __attribute__((weak));
    extern char __ocl_code_end   __attribute__((weak));
};
#endif

namespace __impl { namespace opencl { namespace hpe {

CLBufferPool::~CLBufferPool()
{
    CLMemMap::iterator it;

    for(it = begin(); it != end(); it++) {
        CLMemList::iterator jt;
        CLMemList list = it->second;
        for (jt = list.begin(); jt != list.end(); jt++) {
            cl_int ret;
            ret = clReleaseMemObject(jt->first);
            ASSERTION(ret == CL_SUCCESS);
        }
    }
}

bool
CLBufferPool::getCLMem(size_t size, cl_mem &mem, hostptr_t &addr)
{
    lock(); 

    bool ret = false;

    CLMemMap::iterator it = find(size);

    if (it != end())  {
        CLMemList &list = it->second;
        if (it->second.size() > 0) {
            mem = list.front().first;
            addr = list.front().second;
            list.pop_front();
            ret = true;
        }
    }
    unlock();

    return ret;
}

void CLBufferPool::putCLMem(size_t size, cl_mem mem, hostptr_t addr)
{
    lock(); 

    CLMemMap::iterator it = find(size);

    if (it != end())  {
        CLMemList &list = it->second;
        list.push_back(std::make_pair(mem, addr));
    } else {
        CLMemList list;
        list.push_back(std::make_pair(mem, addr));
        insert(CLMemMap::value_type(size, list));
    }
    unlock();
}

Accelerator::AcceleratorMap *Accelerator::Accelerators_ = NULL;
HostMap *Accelerator::GlobalHostAlloc_;

Accelerator::Accelerator(int n, cl_context context, cl_device_id device) :
    gmac::core::hpe::Accelerator(n), ctx_(context), device_(device)
{
    // Not used for now
    busId_ = 0;
    busAccId_ = 0;

    cl_bool val = CL_FALSE;
    cl_int ret = clGetDeviceInfo(device_, CL_DEVICE_HOST_UNIFIED_MEMORY,
        sizeof(val), NULL, NULL);
    if(ret == CL_SUCCESS) integrated_ = (val == CL_TRUE);
    else integrated_ = false;

    ret = clRetainContext(ctx_);
    CFATAL(ret == CL_SUCCESS, "Unable to retain OpenCL context");

#if 0
    cl_command_queue stream;
    cl_command_queue_properties properties = 0;
#if defined(USE_TRACE)
    properties |= CL_QUEUE_PROFILING_ENABLE;
#endif
    stream = clCreateCommandQueue(ctx_, device_, properties, &ret);
    CFATAL(ret == CL_SUCCESS, "Unable to create OpenCL stream");
    cmd_.add(stream);
#endif
}

Accelerator::~Accelerator()
{
    std::vector<cl_program> &programs = (*Accelerators_)[this];
    std::vector<cl_program>::const_iterator i;
    // We cannot call OpenCL at destruction time because the library
    // might have been unloaded
    for(i = programs.begin(); i != programs.end(); i++) {
        cl_int ret = clReleaseProgram(*i);
        ASSERTION(ret == CL_SUCCESS);
    }
    Accelerators_->erase(this);
    if(Accelerators_->empty()) {
        delete Accelerators_;
        Accelerators_ = NULL;
        delete GlobalHostAlloc_;
        GlobalHostAlloc_ = NULL;
    }

    cl_int ret = CL_SUCCESS;
    ret = clReleaseContext(ctx_);
    ASSERTION(ret == CL_SUCCESS);
}

void Accelerator::init()
{
}

core::hpe::Mode *Accelerator::createMode(core::hpe::Process &proc)
{
    trace::EnterCurrentFunction();
    core::hpe::Mode *mode = ModeFactory::create(dynamic_cast<core::hpe::Process &>(proc), *this);
    if (mode != NULL) {
        registerMode(*mode);
    }
    trace::ExitCurrentFunction();

    TRACE(LOCAL, "Creating Execution Mode %p to Accelerator", mode);
    return mode;
}

gmacError_t Accelerator::map(accptr_t &dst, hostptr_t src, size_t size, unsigned align) 
{
    trace::EnterCurrentFunction();

    cl_int ret = CL_SUCCESS;
    trace::SetThreadState(trace::Wait);
    dst(clCreateBuffer(ctx_, CL_MEM_READ_WRITE, size, NULL, &ret));
    if(ret != CL_SUCCESS) return error(ret);
    trace::SetThreadState(trace::Running);

    allocations_.insert(src, dst, size);

    TRACE(LOCAL, "Allocating accelerator memory (%d bytes) @ %p", size, dst.get());

    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::unmap(hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    ASSERTION(host != NULL);

    accptr_t addr;
    size_t s;

    bool hasMapping = allocations_.find(host, addr, s);
    ASSERTION(hasMapping == true);
    ASSERTION(s == size);
    allocations_.erase(host, size);

    TRACE(LOCAL, "Releasing accelerator memory @ %p", addr.get());

    trace::SetThreadState(trace::Wait);
    cl_int ret = CL_SUCCESS;
    ret = clReleaseMemObject(addr.get());
    trace::SetThreadState(trace::Running);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    TRACE(LOCAL, "Copy to accelerator: %p ("FMT_SIZE") @ %p", host, size, acc.get());
    trace::SetThreadState(trace::Wait);
    cl_event event;
    trace_.init(trace_.getThreadId(), mode.id());
    cl_int ret = clEnqueueWriteBuffer(cmd_.front(), acc.get(),
        CL_TRUE, acc.offset(), size, host, 0, NULL, &event);
    CFATAL(ret == CL_SUCCESS, "Error copying to accelerator: %d", ret);
    trace_.trace(event, event, size);
    trace::SetThreadState(trace::Running);
    cl_int clret = clReleaseEvent(event);
    ASSERTION(clret == CL_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyToHost(hostptr_t host, const accptr_t acc, size_t count, core::hpe::Mode &mode)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    TRACE(LOCAL, "Copy to host: %p ("FMT_SIZE") @ %p", host, count, acc.get());
    trace::SetThreadState(trace::Wait);
    cl_event event;
    trace_.init(mode.id(), trace_.getThreadId());
    cl_int ret = clEnqueueReadBuffer(cmd_.front(), acc.get(),
        CL_TRUE, acc.offset(), count, host, 0, NULL, &event);
    CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
    trace_.trace(event, event, count);
    trace::SetThreadState(trace::Running);
    cl_int clret = clReleaseEvent(event);
    ASSERTION(clret == CL_SUCCESS);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    TRACE(LOCAL, "Copy accelerator-accelerator ("FMT_SIZE") @ %p - %p", size,
        src.get(), dst.get());
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    cl_int ret = clEnqueueReadBuffer(cmd_.front(), src.get(), CL_FALSE,
        src.offset(), size, tmp, 0, NULL, NULL);
    CFATAL(ret == CL_SUCCESS, "Error copying to host: %d", ret);
    if(ret == CL_SUCCESS) {
        ret = clEnqueueWriteBuffer(cmd_.front(), dst.get(), CL_TRUE,
                dst.offset(), size, tmp, 0, NULL, NULL);
        CFATAL(ret == CL_SUCCESS, "Error copying to device: %d", ret);
    }
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::memset(accptr_t addr, int c, size_t size)
{
    trace::EnterCurrentFunction();
    if (cmd_.empty() == true) createCLstream();
    // TODO: This is a very inefficient implementation. We might consider
    // using a kernel for this task
    void *tmp = ::malloc(size);
    ::memset(tmp, c, size);
    cl_int ret = clEnqueueWriteBuffer(cmd_.front() , addr.get(),
        CL_TRUE, addr.offset(), size, tmp, 0, NULL, NULL);
    ::free(tmp);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::sync()
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Waiting for accelerator to finish all activities");
    cl_int ret = cmd_.sync();
    trace::ExitCurrentFunction();
    return error(ret);
}

void
Accelerator::addAccelerator(Accelerator &acc)
{
    std::pair<Accelerator *, std::vector<cl_program> > pair(&acc, std::vector<cl_program>());
    if(Accelerators_ == NULL) {
        Accelerators_ = new AcceleratorMap();
        GlobalHostAlloc_ = new HostMap();
    }
    Accelerators_->insert(pair);
}

Kernel *
Accelerator::getKernel(gmac_kernel_id_t k)
{
    std::vector<cl_program> &programs = (*Accelerators_)[this];
    ASSERTION(programs.size() > 0);

    cl_int err = CL_SUCCESS;
    std::vector<cl_program>::const_iterator i;
    for(i = programs.begin(); i != programs.end(); i++) {
        cl_kernel kernel = clCreateKernel(*i, k, &err);
        if(err != CL_SUCCESS) continue;
        return new Kernel(__impl::core::hpe::KernelDescriptor(k, k), kernel);
    }
    return NULL;
}

gmacError_t Accelerator::prepareEmbeddedCLCode()
{
#ifndef _MSC_VER
    const char *CL_MAGIC = "!@#~";

    trace::EnterCurrentFunction();

    TRACE(GLOBAL, "Preparing embedded code");

    if (&__ocl_code_start != NULL &&
        &__ocl_code_end   != NULL) {
        char *code = &__ocl_code_start;
        char *cursor = strstr(code, CL_MAGIC);

        while (cursor != NULL && cursor < &__ocl_code_end) {
            char *params = cursor + strlen(CL_MAGIC);

            size_t fileSize = cursor - code;
            size_t paramsSize = 0;
            char *file = new char[fileSize + 1];
            char *fileParams = NULL;
            ::memcpy(file, code, fileSize);
            file[fileSize] = '\0';

            cursor = ::strstr(cursor + 1, CL_MAGIC);
            if (cursor != params) {
                paramsSize = cursor - params;
                fileParams = new char[paramsSize + 1];
                ::memcpy(fileParams, params, paramsSize);
                fileParams[paramsSize] = '\0';
            }
            TRACE(GLOBAL, "Compiling file in embedded code");
            gmacError_t ret = prepareCLCode(file, fileParams);
            if (ret != gmacSuccess) {
                abort();
                trace::ExitCurrentFunction();
                return error(ret);
            }

            code = cursor + strlen(CL_MAGIC);
            cursor = ::strstr(cursor + 1, CL_MAGIC);
            if (fileParams != NULL) delete [] fileParams;
            delete [] file;
        }
    }
    trace::ExitCurrentFunction();
#endif
    return gmacSuccess;
}

gmacError_t Accelerator::prepareCLCode(const char *code, const char *flags)
{
    trace::EnterCurrentFunction();

    cl_int ret;
    AcceleratorMap::iterator it;
    for (it = Accelerators_->begin(); it != Accelerators_->end(); it++) {
        cl_program program = clCreateProgramWithSource(
            it->first->ctx_, 1, &code, NULL, &ret);
        if (ret == CL_SUCCESS) {
            // TODO use the callback parameter to allow background code compilation
            ret = clBuildProgram(program, 1, &it->first->device_, flags, NULL, NULL);
        }
        if (ret == CL_SUCCESS) {
            it->second.push_back(program);
            TRACE(GLOBAL, "Compilation OK for accelerator: %d", it->first->device_);
        } else {
            size_t len;
            cl_int ret2 = clGetProgramBuildInfo(program, it->first->device_,
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            ASSERTION(ret2 == CL_SUCCESS);
            char *msg = new char[len + 1];
            ret2 = clGetProgramBuildInfo(program, it->first->device_,
                    CL_PROGRAM_BUILD_LOG, len, msg, NULL);
            ASSERTION(ret2 == CL_SUCCESS);
            msg[len] = '\0';
            TRACE(GLOBAL, "Error compiling code accelerator: %d\n%s",
                it->first->device_, msg);
            delete [] msg;
            break;
        }
    }
    clUnloadCompiler();
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::prepareCLBinary(const unsigned char *binary, size_t size, const char *flags)
{
    trace::EnterCurrentFunction();

    cl_int ret;
    AcceleratorMap::iterator it;
    for (it = Accelerators_->begin(); it != Accelerators_->end(); it++) {
        cl_program program = clCreateProgramWithBinary(it->first->ctx_, 1, &it->first->device_, &size, &binary, NULL, &ret);
        if (ret == CL_SUCCESS) {
            // TODO use the callback parameter to allow background code compilation
            ret = clBuildProgram(program, 1, &it->first->device_, flags, NULL, NULL);
        }
        if (ret == CL_SUCCESS) {
            it->second.push_back(program);
            TRACE(GLOBAL, "Compilation OK for accelerator: %d", it->first->device_);
        } else {
            size_t len;
            cl_int ret2 = clGetProgramBuildInfo(program, it->first->device_, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            ASSERTION(ret2 == CL_SUCCESS);
            char *msg = new char[len + 1];
            ret2 = clGetProgramBuildInfo(program, it->first->device_, CL_PROGRAM_BUILD_LOG, len, msg, NULL);
            ASSERTION(ret2 == CL_SUCCESS);
            msg[len] = '\0';
            TRACE(GLOBAL, "Error compiling code on accelerator %d\n%s", it->first->device_, msg);
            delete [] msg;

            break;
        }
    }
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_command_queue Accelerator::createCLstream()
{
    trace::EnterCurrentFunction();
    cl_command_queue stream;
    cl_int error;
    cl_command_queue_properties prop = 0;
#if defined(USE_TRACE)
    prop |= CL_QUEUE_PROFILING_ENABLE;
#endif
    stream = clCreateCommandQueue(ctx_, device_, prop, &error);
    CFATAL(error == CL_SUCCESS, "Unable to create OpenCL stream");
    TRACE(LOCAL, "Created OpenCL stream %p, in Accelerator %p", stream, this);
    cmd_.add(stream);
    trace::ExitCurrentFunction();
    return stream;
}

void Accelerator::destroyCLstream(cl_command_queue stream)
{
    trace::EnterCurrentFunction();
	// We cannot remove the command queue because the OpenCL DLL might
	// have been already unloaded
    cl_int ret = CL_SUCCESS;
    ret = clReleaseCommandQueue(stream);
    CFATAL(ret == CL_SUCCESS, "Unable to destroy OpenCL stream");

    TRACE(LOCAL, "Destroyed OpenCL stream %p, in Accelerator %p", stream, this);
    cmd_.remove(stream);
    trace::ExitCurrentFunction();
}


gmacError_t Accelerator::syncCLstream(cl_command_queue stream)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Waiting for stream %p in Accelerator %p", stream, this);
    cl_int ret = clFinish(stream);
    CFATAL(ret == CL_SUCCESS, "Error syncing cl_command_queue: %d", ret);
    trace::ExitCurrentFunction();
    return error(ret);
}

cl_int Accelerator::queryCLevent(cl_event event)
{
    cl_int ret = CL_SUCCESS;
    trace::EnterCurrentFunction();
    cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS,
        sizeof(cl_int), &ret, NULL);
    CFATAL(err == CL_SUCCESS, "Error querying cl_event: %d", err);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Accelerator::syncCLevent(cl_event event)
{
    trace::EnterCurrentFunction();
    TRACE(LOCAL, "Accelerator waiting for event");
#if defined(OPENCL_1_1)
	cl_int status = 0;
	cl_int ret = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);	
	while(ret == CL_SUCCESS && status > 0) {
		ret = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status, NULL);
	}
#else
	cl_int ret = clWaitForEvents(1, &event);
#endif
    CFATAL(ret == CL_SUCCESS, "Error syncing cl_event: %d", ret);
    trace::ExitCurrentFunction();
    return error(ret);
}

gmacError_t Accelerator::timeCLevents(uint64_t &t, cl_event start, cl_event end)
{
    uint64_t startTime, endTime;
    t = 0;
    cl_int ret = clGetEventProfilingInfo(start, CL_PROFILING_COMMAND_START,
        sizeof(startTime), &startTime, NULL);
    if(ret != CL_SUCCESS) return error(CL_SUCCESS);
    ret = clGetEventProfilingInfo(end, CL_PROFILING_COMMAND_END,
        sizeof(endTime), &endTime, NULL);
    if(ret != CL_SUCCESS) return error(CL_SUCCESS);
    t = (endTime - startTime) / 1000;
    return error(ret);
}


gmacError_t Accelerator::hostAlloc(hostptr_t &addr, size_t size)
{
    // There is not reliable way to get zero-copy memory
    addr = NULL;
    return gmacErrorMemoryAllocation;
#if 0
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    addr = NULL;

    // Get a memory object in the host memory
    cl_mem mem = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            size, NULL, &ret);
    if(ret == CL_SUCCESS) {
        // Get the host pointer for the memory object
        addr = (hostptr_t)clEnqueueMapBuffer(cmd_.front(), mem, CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);
        // Insert the object in the allocation map for the accelerator
        if(ret != CL_SUCCESS) clReleaseMemObject(mem);
        else {
            localHostAlloc_.insert(addr, mem, size);
            Accelerator::GlobalHostAlloc_->insert(addr, mem, size);
        }
    }
    trace::ExitCurrentFunction();
    return error(ret);
#endif
}

gmacError_t Accelerator::allocCLBuffer(cl_mem &mem, hostptr_t &addr, size_t size)
{
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;

    if (clMem_.getCLMem(size, mem, addr)) {
        goto exit;
    }
    if (cmd_.empty() == true) createCLstream();

    // Get a memory object in the host memory
    mem = clCreateBuffer(ctx_, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
            size, NULL, &ret);
    if(ret == CL_SUCCESS) {
        // Get the host pointer for the memory object
        addr = (hostptr_t)clEnqueueMapBuffer(cmd_.front(), mem, CL_TRUE,
                CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL, &ret);
        // Insert the object in the allocation map for the accelerator
        if(ret != CL_SUCCESS) clReleaseMemObject(mem);
    }

exit:
    trace::ExitCurrentFunction();
    return error(ret);
}


gmacError_t Accelerator::hostFree(hostptr_t addr)
{
    // There is not reliable way to get zero-copy memory
    return gmacErrorMemoryAllocation;
#if 0
    trace::EnterCurrentFunction();
    cl_int ret = CL_SUCCESS;
    cl_mem mem = NULL;
    size_t size;
    if(localHostAlloc_.translate(addr, mem, size) == true) {
        localHostAlloc_.remove(addr); 
        Accelerator::GlobalHostAlloc_->remove(addr);
        ret = clReleaseMemObject(mem);
    }
    
    trace::ExitCurrentFunction();
    return error(ret);
#endif
}

gmacError_t Accelerator::freeCLBuffer(cl_mem mem, hostptr_t addr, size_t size)
{
    trace::EnterCurrentFunction();
    clMem_.putCLMem(size, mem, addr);
    trace::ExitCurrentFunction();

    return error(CL_SUCCESS);
}

accptr_t Accelerator::hostMapAddr(const hostptr_t addr)
{
    // There is not reliable way to get zero-copy memory
    return accptr_t(0);
#if 0
    cl_mem device;
    size_t size = 0;
    if(localHostAlloc_.translate(addr, device, size) == false) {
        CFATAL(Accelerator::GlobalHostAlloc_->translate(addr, device, size) == false,
               "Error translating address %p", (void *) addr);
    }
    return accptr_t(device, 0);
#endif
}

void Accelerator::memInfo(size_t &free, size_t &total) const
{
    cl_int ret = CL_SUCCESS;
    cl_ulong value = 0;
    ret = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_SIZE,
        sizeof(value), &value, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    total = size_t(value);

    // TODO: This is actually wrong, but OpenCL do not let us know the
    // amount of free memory in the accelerator
    ret = clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
        sizeof(value), &value, NULL);
    CFATAL(ret == CL_SUCCESS , "Unable to get attribute %d", ret);
    free = size_t(value);
}

}}}
