#ifdef USE_DBC

#include "IOBuffer.h"
#include "core/IOBuffer.h"

namespace __dbc { namespace core {


IOBuffer::IOBuffer(void *addr, size_t size):__impl::core::IOBuffer(addr, size)
{
}

IOBuffer::~IOBuffer()
{
}
void IOBuffer::lock()
{
 
      __impl::core:IOBuffer::lock();
 
     //thread_mutex_lock(&internal_);
     //ENSURES(owner == 0);
     //ENSURES(locked == false);
     //locked_ = true;
     //owner_ = pthread_self();
     //pthread_mutex_unlock(&internal_);



}

void IOBuffer::unlock()
{
 
     // pthread_mutex_lock(&internal_);
     // REQUIRES(locked_ == true);
     // REQUIRES(owner_ == pthread_self());
     // owner_ = NULL;
     // locked_ = false;

      __impl::core::IOBuffer::unlock();  

     // pthread_mutex_unlock(&internal_);

}

}}
#endif 

