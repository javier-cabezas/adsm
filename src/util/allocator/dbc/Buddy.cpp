#if defined(USE_DBC)

#include "Buddy.h"

namespace __dbc { namespace core { namespace allocator {

Buddy::Buddy(hostptr_t addr, size_t size) :
    __impl::core::allocator::Buddy(addr, size)
{}

off_t Buddy::getFromList(uint8_t i)
{
    REQUIRES(i >= 0);
    return __impl::core::allocator::Buddy::getFromList(i);
}

void Buddy::putToList(off_t addr, uint8_t i)
{
    REQUIRES(addr >= 0 && size_t(addr) < size_);
    REQUIRES(i >= 0);
    return __impl::core::allocator::Buddy::putToList(addr, i);
}

hostptr_t Buddy::get(size_t &size)
{
    REQUIRES(size > 0);
    hostptr_t ret = __impl::core::allocator::Buddy::get(size);
    ENSURES(ret == NULL || (ret >= addr_ && ret <= (addr_ + size_ - size)));
    return ret;
}

void Buddy::put(hostptr_t addr, size_t size)
{
    REQUIRES(addr >= addr_ && addr <= (addr_ + size_ - size));
    REQUIRES(size > 0);
    __impl::core::allocator::Buddy::put(addr, size);
}

}}}

#endif
