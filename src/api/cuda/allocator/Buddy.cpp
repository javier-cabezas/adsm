#include "core/allocator/Buddy.h"
#include "../Mode.h"

#include "util/Logger.h"

namespace gmac { namespace kernel { namespace allocator {

void Buddy::initMemory()
{
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    gmacError_t ret = mode.hostAlloc(&addr_, size_);
    assertion(ret == gmacSuccess);
    _tree[index_].push_back(0);
    trace("Getting pinned memory for I/O operations at %p", addr_);
}

void Buddy::finiMemory()
{
    trace("Releasing pinned memory for I/O operations");
    gmac::cuda::Mode &mode = gmac::cuda::Mode::current();
    gmacError_t ret = mode.hostFree(addr_);
    assertion(ret == gmacSuccess);
}


}}}
