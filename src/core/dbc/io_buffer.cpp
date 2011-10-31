#ifdef USE_DBC

#include "core/dbc/io_buffer.h"

namespace __dbc { namespace core {


io_buffer::io_buffer(__impl::hal::context_t &context, size_t size, GmacProtection prot) :
    __impl::core::io_buffer(context, size, prot)
{
    // This check goes out because OpenCL will always use 0 as base address
    REQUIRES(size > 0);
}

io_buffer::~io_buffer()
{
}

uint8_t *io_buffer::addr() const
{
    uint8_t *ret = __impl::core::io_buffer::addr();
    ENSURES(ret != NULL);
    return ret;
}

uint8_t *io_buffer::end() const
{
    uint8_t *ret = __impl::core::io_buffer::end();
    ENSURES(ret !=  NULL);
    return ret;
}

size_t io_buffer::size() const
{
    size_t ret = __impl::core::io_buffer::size();
    ENSURES(ret > 0);
    return ret;
}

}}
#endif 

