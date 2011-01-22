#ifndef GMAC_CORE_DBC_IOBUFFER_H_
#define GMAC_CORE_DBC_IOBUFFER_H_

#include "dbc/types.h"
#include "dbc/Contract.h"

#include "pthread.h"
#include "core/IOBuffer.h"

namespace __dbc { namespace core {

class GMAC_LOCAL IOBuffer: 
    public __impl::core::IOBuffer, 
    public virtual Contract {
    DBC_TESTED(__impl::core::IOBuffer)

protected:
    pthread_mutex_t internal_;
    pthread_t owner_; 
    bool locked_; 

protected: 
    IOBuffer(void *addr, size_t size);

public:
    virtual ~IOBuffer();
    void lock();
    void unlock();
};
      
}}
#endif


