#ifndef __MANAGER_IPP_
#define __MANAGER_IPP_

namespace gmac { namespace memory {

inline Manager::~Manager()
{
    trace("Memory manager finishes");
    assertion(__count == 0);
    delete protocol;
}


}};
#endif
