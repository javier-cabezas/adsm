#ifdef USE_DBC

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
    __impl::core::IOBuffer::lock();

    pthread_mutex_lock(&internal_);
    ENSURES(owner_ == 0);
    ENSURES(locked_ == false);
    locked_ = true;
    owner_ = pthread_self();
    pthread_mutex_unlock(&internal_);



}

void IOBuffer::unlock()
{
 
      pthread_mutex_lock(&internal_);
      REQUIRES(locked_ == true);
      REQUIRES(owner_ == pthread_self());
      owner_ = NULL;
      locked_ = false;

      __impl::core::IOBuffer::unlock();  

      pthread_mutex_unlock(&internal_);

}

}}
#endif 

