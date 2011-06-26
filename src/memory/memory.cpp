#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#include "memory/BlockGroup.h"
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
unsigned SubBlocks_;
size_t SubBlockSize_;
unsigned BlockShift_;
unsigned SubBlockShift_;
long_t SubBlockMask_;
#endif


//CONSTRUCTOR(init);
void Init()
{
    BlockSize_     = util::params::ParamBlockSize;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    SubBlockSize_  = util::params::ParamSubBlockSize;
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
    if(strcasecmp(util::params::ParamProtocol, "Rolling") == 0 ||
       strcasecmp(util::params::ParamProtocol, "Lazy") == 0) {
        size_t rollSize;
        if(strcasecmp(util::params::ParamProtocol, "Rolling") == 0) {
            rollSize = util::params::ParamRollSize;
        } else {
            rollSize = (size_t)-1;
        }
#define USE_GENERIC_OBJECTS 0
        if(0 != (flags & 0x1)) {
#if USE_GENERIC_OBJECTS == 1
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(unsigned(rollSize));
#else
            ret = new gmac::memory::protocol::Lazy<
                DistributedObject<protocol::lazy::BlockState> >(unsigned(rollSize));
#endif
        } else {
#if USE_GENERIC_OBJECTS == 1
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(unsigned(rollSize));
#else
            ret = new gmac::memory::protocol::Lazy<
                gmac::memory::SharedObject<protocol::lazy::BlockState> >(unsigned(rollSize));
#endif
        }
    }
#ifdef USE_VM
    else if(strcasecmp(util::params::ParamProtocol, "Gather") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new gmac::memory::protocol::Lazy<
                DistributedObject<protocol::lazy::BlockState> >(
                (size_t)-1);
        }
        else {
            ret = new gmac::memory::protocol::Lazy<
                gmac::memory::SharedObject<protocol::lazy::BlockState> >(
                (size_t)-1);
        }
    }
#endif
    else {
        FATAL("Memory Coherence Protocol not defined");
    }
    return ret;
}


}}
