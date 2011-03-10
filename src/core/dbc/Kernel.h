#ifndef GMAC_CORE_DBC_KERNEL_H_
#define GMAC_CORE_DBC_KERNEL_H_

#include "core/Kernel.h"
#include "dbc/Contract.h"
namespace __dbc { namespace core {

class  GMAC_LOCAL Kernel : 
    public __impl::core::Kernel,
    public virtual Contract { 
    DBC_TESTED(__impl::core::Kernel)
public:
   Kernel(const __impl::core::KernelDescriptor& k);
   virtual ~Kernel();
};
}}

#endif  
  
