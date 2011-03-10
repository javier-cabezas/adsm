#ifdef USE_DBC

#include "core/Kernel.h"

namespace  __dbc {namespace core {

Kernel::Kernel(const __impl::core::KernelDescriptor& k) : __impl::core::Kernel(k)
{
       REQUIRES(&k != NULL);
}

Kernel::~Kernel()
{
}

}}
#endif 

  

