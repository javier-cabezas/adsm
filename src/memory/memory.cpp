#include "manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#include "memory/BlockGroup.h"

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
unsigned SubBlocks_;
size_t SubBlockSize_;
unsigned BlockShift_;
unsigned SubBlockShift_;
long_t SubBlockMask_;
#endif


//CONSTRUCTOR(init);
void Init()
{
    BlockSize_     = config::params::BlockSize;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    SubBlockSize_  = config::params::SubBlockSize;
    SubBlocks_     = BlockSize_/SubBlockSize_;
    BlockShift_    = (unsigned) log2(BlockSize_);
    SubBlockShift_ = (unsigned) log2(SubBlockSize_);
    SubBlockMask_  = SubBlocks_ - 1;

#if defined(USE_VM)
    // TODO: Remove static initialization
    vm::Bitmap::Init();
#endif
#endif
}

Protocol *ProtocolInit(unsigned flags)
{
    TRACE(GLOBAL, "Initializing Memory Protocol");
    Protocol *ret = NULL;
    if(strcasecmp(config::params::Protocol, "Rolling") == 0 ||
       strcasecmp(config::params::Protocol, "Lazy") == 0) {
        bool eager;
        if(strcasecmp(config::params::Protocol, "Rolling") == 0) {
            eager = true;
        } else {
            eager = false;
        }
        if(0 != (flags & 0x1)) {
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
        } else {
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
        }
    }
#ifdef USE_VM
    else if(strcasecmp(config::params::Protocol, "Gather") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
        }
        else {
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
        }
    }
#endif
    else {
        FATAL("Memory Coherence Protocol not defined");
    }
    return ret;
}


}}
