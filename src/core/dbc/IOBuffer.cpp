#ifdef USE_DBC

#include "core/IOBuffer.h"

namespace __dbc { namespace core {


IOBuffer::IOBuffer(void *addr, size_t size):__impl::core::IOBuffer(addr, size)
{
}

IOBuffer::~IOBuffer()
{
}

}}
#endif 

