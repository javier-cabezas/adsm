#if defined(USE_DBC)

#include "Buddy.h"

namespace __dbc { namespace util { namespace allocator {

buddy::buddy(hostptr_t addr, size_t size) :
    __impl::util::allocator::buddy(addr, size)
{}

off_t buddy::getFromList(uint8_t i)
{
    return __impl::util::allocator::buddy::getFromList(i);
}

void buddy::putToList(off_t addr, uint8_t i)
{
    REQUIRES(addr >= 0 && size_t(addr) < size_);
    return __impl::util::allocator::buddy::putToList(addr, i);
}

hostptr_t buddy::get(size_t &size)
{
    REQUIRES(size > 0);
    hostptr_t ret = __impl::util::allocator::buddy::get(size);
    ENSURES(ret == NULL || (ret >= addr_ && ret <= (addr_ + size_ - size)));
    return ret;
}

void buddy::put(hostptr_t addr, size_t size)
{
    REQUIRES(addr >= addr_ && addr <= (addr_ + size_ - size));
    REQUIRES(size > 0);
    __impl::util::allocator::buddy::put(addr, size);
}

}}}

#endif
