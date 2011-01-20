#ifndef GMAC_CORE_DBC_ACCELERATOR_H_
#define GMAC_CORE_DBC_ACCELERATOR_H_ 


#include "dbc/types.h"
#include "dbc/Contract.h"
#include "config/config.h"
#include "core/Accelerator.h"
#include "core/Mode.h"




namespace __dbc { namespace core {

class GMAC_LOCAL Accelerator : public __impl::core::Accelerator, 
                               public virtual Contract {
                               DBC_TESTED(__impl::core::Accelerator)
public: 
   Accelerator(int n);
   virtual ~Accelerator();
   void registerMode(__impl::core::Mode&  mode);
   void unregisterMode(__impl::core::Mode& mode);
   
   


 };
}}

#endif 
  
