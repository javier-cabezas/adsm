#ifndef GMAC_CORE_DBC_KERNEL_H_
#define GMAC_CORE_DBC_KERNEL_H_

#include "dbc/Contract.h"
namespace __dbc { namespace core {

class  GMAC_LOCAL Kernel : 
    public __impl::core::Kernel,
    public virtual Contract { 
    DBC_TESTED(__impl::core::Kernel)
public:
   Kernel(const KernelDescriptor & k);
   virtual ~Kernel();
};
 
}}

#endif  
  
