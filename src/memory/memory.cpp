#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"
#include "protocol/Lazy.h"

#if defined(__GNUC__)
#include <strings.h>
#elif defined(_MSC_VER)
#define strcasecmp _stricmp
#endif

namespace __impl { 

namespace core {

void memoryInit(void)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    memory::Manager::create<gmac::memory::Manager>();
    memory::Allocator::create<__impl::memory::allocator::Slab>();
}

memory::Protocol *protocolInit(unsigned flags)
{
    TRACE(GLOBAL, "Initializing Memory Protocol");
    memory::Protocol *ret = NULL;
    if(strcasecmp(paramProtocol, "Rolling") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new memory::protocol::Lazy<
                memory::DistributedObject<memory::protocol::LazyBase::State> >(
                paramRollSize);
        }
        else {
            ret = new memory::protocol::Lazy<
                gmac::memory::SharedObject<memory::protocol::LazyBase::State> >(
                paramRollSize);
        }
    }
    else if(strcasecmp(paramProtocol, "Lazy") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new memory::protocol::Lazy<
                memory::DistributedObject<memory::protocol::LazyBase::State> >(
                (size_t)-1);
        }
        else {
            ret = new memory::protocol::Lazy<
                gmac::memory::SharedObject<memory::protocol::LazyBase::State> >(
                (size_t)-1);
        }
    }
    else {
        FATAL("Memory Coherence Protocol not defined");
    }
    return ret;
}

void memoryFini(void)
{
	TRACE(GLOBAL, "Cleaning Memory Subsystem");
    memory::Allocator::destroy();
    memory::Manager::destroy();
}


}}
