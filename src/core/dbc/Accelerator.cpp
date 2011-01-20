#ifdef USE_DBC

#include "core/Accelerator.h"
#include "core/Mode.h"

namespace __dbc { namespace core  {
    
Accelerator::Accelerator(int n) : __impl::core::Accelerator(n)
{
}

Accelerator::~Accelerator()
{
}

void Accelerator::registerMode(__impl::core::Mode& mode)
{
   REQUIRES(&mode != NULL);
  //gmacError_t ret;
    __impl::core::Accelerator::registerMode(mode);
}

void Accelerator::unregisterMode(__impl::core::Mode& mode)
{
   REQUIRES(&mode != NULL);       
  // gmacError_t ret; 
    __impl::core::Accelerator::unregisterMode(mode);
}
 
/*
// Declarations of  those method  __impl namespace  in __dbc namespace 
inline unsigned  __impl::core::Accelerator::load() const; 
inline unsigned  __impl::core::Accelerator::id() const; 
inline unsigned  __impl::core::Accelerator::busId_() const;
inline unsigned  __impl::core::Accelerator::busAccId() const;
inline bool __impl::core::Accelerator::integrated() const;
*/

}}
#endif

