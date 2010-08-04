#ifndef __MANAGER_IPP_
#define __MANAGER_IPP_

namespace gmac { namespace memory {

Manager::~Manager()
{
    trace("Memory manager finishes");
    assertion(__count == 0);
    delete __protocol;
}



}};
#endif
