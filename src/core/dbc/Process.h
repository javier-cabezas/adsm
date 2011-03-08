#ifndef GMAC_CORE_DBC_PROCESS_H_ 
#define GMAC_CORE_DBC_PROCESS_H_

#include "dbc/Contract.h"

namespace __dbc { namespace core {
   
class GMAC_LOCAL Process :
   public __impl::core::Process,
   public virtual Contract {
   DBC_TESTED(__impl::core::Process)
protected:
    Process();
public:
    virtual ~Process();
    void initThread();
    void finiThread();
#define ACC_AUTO_BIND -1 
    Mode * createMode(int acc =ACC_AUTO_BIND);
    void removeMode(Mode &mode);
//    gmacError_t globalMalloc(memory::Object  &Object);
//    gmacError_t globalFree(memory::Object &Object);
    

};
}}

#endif


    
  
