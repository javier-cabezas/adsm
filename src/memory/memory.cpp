#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#include "protocol/Lazy.h"

#if defined(__GNUC__)
#include <strings.h>
#elif defined(_MSC_VER)
#define strcasecmp _stricmp
#endif

namespace __impl { namespace core {

void memoryInit(void)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    memory::Manager::create<gmac::memory::Manager>();
    memory::Allocator::create<__impl::memory::allocator::Slab>();
}

memory::Protocol *protocolInit(void)
{
    TRACE(GLOBAL, "Initializing Memory Protocol");
    memory::Protocol *ret = NULL;
    if(strcasecmp(paramProtocol, "Rolling") == 0) {
        ret = new memory::protocol::Lazy((unsigned)paramRollSize);
    }
    else if(strcasecmp(paramProtocol, "Lazy") == 0) {
        ret = new memory::protocol::Lazy((unsigned)-1);
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
