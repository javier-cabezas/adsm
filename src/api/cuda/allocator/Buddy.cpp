#include <kernel/allocator/Buddy.h>
#include "../Mode.h"

#include <util/Logger.h>

namespace gmac { namespace kernel { namespace allocator {

void Buddy::initMemory()
{
    gmac::cuda::Mode *mode = dynamic_cast<gmac::cuda::Mode *>(gmac::Mode::current());
    gmacError_t ret = mode->hostAlloc(&_addr, _size);
    gmac::util::Logger::ASSERTION(ret = gmacSuccess);
}

void Buddy::finiMemory()
{
    gmac::cuda::Mode *mode = dynamic_cast<gmac::cuda::Mode *>(gmac::Mode::current());
    gmacError_t ret = mode->hostFree(&_addr);
    gmac::util::Logger::ASSERTION(ret = gmacSuccess);
}


}}}
