#include "memory/Manager.h"
#include "trace/Tracer.h"

#include "Accelerator.h"
#include "Context.h"

namespace __impl { namespace core {

Context::Context(Accelerator &acc, unsigned id) :
    gmac::util::RWLock("Context"),
    acc_(acc),
    id_(id)
{
}

Context::~Context()
{ 
}

void
Context::init()
{
}

gmacError_t Context::copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size)
{
    trace::EnterCurrentFunction();
    uint64_t start, end;
    trace::TimeMark(start);
    gmacError_t ret = acc_.copyToAccelerator(acc, host, size);
    trace::TimeMark(end);
    trace::DataCommunication(util::GetThreadId(), id_, end - start, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyToHost(hostptr_t host, const accptr_t acc, size_t size)
{
    trace::EnterCurrentFunction();
    uint64_t start, end;
    trace::TimeMark(start);
    gmacError_t ret = acc_.copyToHost(host, acc, size);
    trace::TimeMark(end);
    trace::DataCommunication(id_, util::GetThreadId(), end - start, size);
    trace::ExitCurrentFunction();
    return ret;
}

gmacError_t Context::copyAccelerator(accptr_t dst, const accptr_t src, size_t size)
{
    trace::EnterCurrentFunction();
    gmacError_t ret = acc_.copyAccelerator(dst, src, size);
    trace::ExitCurrentFunction();
    return ret;
}

}}
