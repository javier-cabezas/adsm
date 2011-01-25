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

namespace memory {

size_t BlockSize_;
#ifdef USE_VM
size_t SubBlockSize_;
unsigned long BlockShift_;
unsigned long SubBlockShift_;
unsigned long SubBlockMask_;
#endif

void Init(void)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    Manager::create<gmac::memory::Manager>();
    Allocator::create<__impl::memory::allocator::Slab>();

    BlockSize_     = util::params::ParamBlockSize;
#ifdef USE_VM
    SubBlockSize_  = util::params::ParamBlockSize/util::params::ParamSubBlocks;
    BlockShift_    = ceilf(log2f(float(util::params::ParamBlockSize)));
    SubBlockShift_ = ceilf(log2f(float(util::params::ParamBlockSize/util::params::ParamSubBlocks)));
    SubBlockMask_  = util::params::ParamSubBlocks - 1;

    vm::Bitmap::Init();
#endif
}

Protocol *ProtocolInit(unsigned flags)
{
    TRACE(GLOBAL, "Initializing Memory Protocol");
    Protocol *ret = NULL;
    if(strcasecmp(util::params::ParamProtocol, "Rolling") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new protocol::Lazy<
                DistributedObject<protocol::LazyBase::State> >(
                util::params::ParamRollSize);
        }
        else {
            ret = new protocol::Lazy<
                gmac::memory::SharedObject<protocol::LazyBase::State> >(
                util::params::ParamRollSize);
        }
    }
    else if(strcasecmp(util::params::ParamProtocol, "Lazy") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new protocol::Lazy<
                DistributedObject<protocol::LazyBase::State> >(
                (size_t)-1);
        }
        else {
            ret = new protocol::Lazy<
                gmac::memory::SharedObject<protocol::LazyBase::State> >(
                (size_t)-1);
        }
    }
    else {
        FATAL("Memory Coherence Protocol not defined");
    }
    return ret;
}

void Fini(void)
{
	TRACE(GLOBAL, "Cleaning Memory Subsystem");
    Allocator::destroy();
    Manager::destroy();
}

}}
