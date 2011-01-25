#ifndef GMAC_CORE_DBC_IOBUFFER_H_
#define GMAC_CORE_DBC_IOBUFFER_H_

#include "dbc/types.h"
#include "dbc/Contract.h"

#include "core/IOBuffer.h"

namespace __dbc { namespace core {

class GMAC_LOCAL IOBuffer: 
    public __impl::core::IOBuffer, 
    public virtual Contract {
    DBC_TESTED(__impl::core::IOBuffer)

protected: 
    IOBuffer(void *addr, size_t size);

public:
    virtual ~IOBuffer();
};
      
}}
#endif


