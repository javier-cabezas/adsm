#include "Manager.h"
#include "Allocator.h"

#include "allocator/Slab.h"

#define USE_GENERIC_OBJECTS 1

#if USE_GENERIC_OBJECTS == 1
#include "memory/BlockGroup.h"
#else
#include "memory/SharedObject.h"
#include "memory/DistributedObject.h"
#endif

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
        bool eager;
        if(strcasecmp(util::params::ParamProtocol, "Rolling") == 0) {
            eager = true;
        } else {
            eager = false;
        }
        if(0 != (flags & 0x1)) {
#if USE_GENERIC_OBJECTS == 1
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
#else
            ret = new gmac::memory::protocol::Lazy<
                DistributedObject<protocol::lazy::BlockState> >(eager);
#endif
        } else {
#if USE_GENERIC_OBJECTS == 1
            ret = new gmac::memory::protocol::Lazy<
                memory::BlockGroup<protocol::lazy::BlockState> >(eager);
#else
            ret = new gmac::memory::protocol::Lazy<
                gmac::memory::SharedObject<protocol::lazy::BlockState> >(eager);
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
