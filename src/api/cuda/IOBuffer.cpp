#include "IOBuffer.h"
#include "Mode.h"

namespace gmac { namespace gpu {

IOBuffer::IOBuffer(size_t size) :
    gmac::IOBuffer(size),
    pin(false)
{
    gmacError_t ret = mode->hostAlloc(&__addr, size);
    if(ret == gmacSuccess) pin = true;
    else {
        __addr = malloc(size);
    }
}

IOBuffer::~IOBuffer()
{
    if(__addr == NULL) return;
    if(pin) mode->hostFree(__addr);
    else free(__addr);
    __addr = NULL;
}


}}

