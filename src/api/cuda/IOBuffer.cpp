#include "IOBuffer.h"
#include "Mode.h"

namespace gmac { namespace gpu {

IOBuffer::IOBuffer(Mode *mode, size_t size) :
    gmac::IOBuffer(mode, size),
    mode(mode)
{
    // Parent class might be already allocated some memory for us
    if(__addr != NULL) free(__addr);
    gmacError_t ret = mode->hostAlloc(&__addr, size);
    if(ret != gmacSuccess) __addr = NULL;
}

IOBuffer::~IOBuffer()
{
    if(__addr == NULL) return;
    mode->hostFree(__addr);
    __addr = NULL;
}


gmacError_t IOBuffer::dump(void *addr, size_t len)
{
    gmacError_t ret;
    lock();
    if(__state != Idle) ret = sync();
    if(ret != gmacSuccess) { unlock(); return ret; }
    __state = Dump;
    ret = mode->bufferToDevice(this, addr, len);
    unlock();
    return ret;
}

gmacError_t IOBuffer::fill(void *addr, size_t len)
{
    gmacError_t ret;
    lock();
    if(__state != Idle) ret = sync();
    if(ret != gmacSuccess) { unlock(); return ret; }
    __state = Fill;
    ret = mode->bufferToHost(this, addr, len);
    unlock();
    return ret;
}

gmacError_t IOBuffer::sync()
{
    gmacError_t ret = gmacSuccess;
    switch(__state) {
        case Dump: ret = mode->waitDevice(); break;
        case Fill: ret = mode->waitHost(); break;
        case Idle: break;
    }     
    __state = Idle;
    return ret;
}


}}

