#include <kernel/allocator/Buddy.h>
#include "../Mode.h"

#include <util/Logger.h>

namespace gmac { namespace kernel { namespace allocator {

void Buddy::initMemory()
{
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    gmacError_t ret = mode.hostAlloc(&_addr, _size);
    assertion(ret == gmacSuccess);
    _tree[_index].push_back(0);
    trace("Getting pinned memory for I/O operations at %p", _addr);
}

void Buddy::finiMemory()
{
    trace("Releasing pinned memory for I/O operations");
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    gmacError_t ret = mode.hostFree(_addr);
    assertion(ret == gmacSuccess);
}


}}}
