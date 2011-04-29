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
unsigned SubBlocks_;
size_t SubBlockSize_;
unsigned BlockShift_;
unsigned SubBlockShift_;
long_t SubBlockMask_;
#endif


CONSTRUCTOR(init);
static void init()
{
    BlockSize_     = util::params::ParamBlockSize;
#if defined(USE_VM) || defined(USE_SUBBLOCK_TRACKING)
    SubBlocks_     = util::params::ParamSubBlocks;
    SubBlockSize_  = util::params::ParamBlockSize/util::params::ParamSubBlocks;
    BlockShift_    = (unsigned) log2(util::params::ParamBlockSize);
    SubBlockShift_ = (unsigned) log2(util::params::ParamBlockSize/util::params::ParamSubBlocks);
    SubBlockMask_  = util::params::ParamSubBlocks - 1;

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
        if(0 != (flags & 0x1)) {
            ret = new protocol::Lazy<
                DistributedObject<protocol::lazy::BlockState> >(rollSize);
        }
        else {
            ret = new protocol::Lazy<
                gmac::memory::SharedObject<protocol::lazy::BlockState> >(rollSize);
        }
    }
#ifdef USE_VM
    else if(strcasecmp(util::params::ParamProtocol, "Gather") == 0) {
        if(0 != (flags & 0x1)) {
            ret = new protocol::Lazy<
                DistributedObject<protocol::lazy::BlockState> >(
                (size_t)-1);
        }
        else {
            ret = new protocol::Lazy<
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
