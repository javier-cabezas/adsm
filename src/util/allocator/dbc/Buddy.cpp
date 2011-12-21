#if defined(USE_DBC)

#include "Buddy.h"

namespace __dbc { namespace util { namespace allocator {

buddy::buddy(host_ptr addr, size_t size) :
    __impl::util::allocator::buddy(addr, size)
{}

off_t buddy::get_from_list(uint8_t i)
{
    return __impl::util::allocator::buddy::get_from_list(i);
}

void buddy::put_to_list(off_t addr, uint8_t i)
{
    REQUIRES(addr >= 0 && size_t(addr) < size_);
    return __impl::util::allocator::buddy::put_to_list(addr, i);
}

host_ptr buddy::get(size_t &size)
{
    REQUIRES(size > 0);
    host_ptr ret = __impl::util::allocator::buddy::get(size);
    ENSURES(ret == NULL || (ret >= addr_ && ret <= (addr_ + size_ - size)));
    return ret;
}

void buddy::put(host_ptr addr, size_t size)
{
    REQUIRES(addr >= addr_ && addr <= (addr_ + size_ - size));
    REQUIRES(size > 0);
    __impl::util::allocator::buddy::put(addr, size);
}

}}}

#endif
