#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"

#ifdef USE_VM
//#include "protocol/Gather.h"
#endif
#include "protocol/Lazy.h"

#if defined(__GNUC__)
#include <strings.h>
#elif defined(_MSC_VER)
#define strcasecmp _stricmp
#endif

namespace __impl { 

namespace memory {

size_t BlockSize_;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
size_t SubBlockSize_;
unsigned BlockShift_;
unsigned SubBlockShift_;
long_t SubBlockMask_;
#endif

void Init(void)
{
	TRACE(GLOBAL, "Initializing Memory Subsystem");
    Manager::create<gmac::memory::Manager>();
    Allocator::create<__impl::memory::allocator::Slab>();

    BlockSize_     = util::params::ParamBlockSize;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    SubBlockSize_  = util::params::ParamBlockSize/util::params::ParamSubBlocks;
    BlockShift_    = (unsigned) log2(util::params::ParamBlockSize);
    SubBlockShift_ = (unsigned) log2(util::params::ParamBlockSize/util::params::ParamSubBlocks);
    SubBlockMask_  = util::params::ParamSubBlocks - 1;

#if defined(USE_VM)
    vm::Bitmap::Init();
#endif
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
#ifdef USE_VM
    else if(strcasecmp(util::params::ParamProtocol, "Gather") == 0) {
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
#endif
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
