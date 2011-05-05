#include "libs/common.h"

#include "util/Atomics.h"
#include "util/Lock.h"
#include "util/Private.h"

class GMAC_LOCAL GMACLock : public gmac::util::RWLock {
public:
    GMACLock() : gmac::util::RWLock("Process") {}

    void lockRead()  const { gmac::util::RWLock::lockRead();  }
    void lockWrite() const { gmac::util::RWLock::lockWrite(); }
    void unlock()    const { gmac::util::RWLock::unlock();   }
};

static __impl::util::Private<const char> inGmac_;

static const char gmacCode = 1;
static const char userCode = 0;

static Atomic gmacInit__ = 0;

CONSTRUCTOR(init);
static void init(void)
{
    /* Create GMAC enter lock and set GMAC as initialized */
    __impl::util::Private<const char>::init(inGmac_);
}
    
void enterGmac()
{
	if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) initGmac();
    inGmac_.set(&gmacCode);
}


void enterGmacExclusive()
{
    inGmac_.set(&gmacCode);
}

void exitGmac()
{
    inGmac_.set(&userCode);
}

char inGmac()
{ 
    if(gmacInit__ == 0) return 1;
    char *ret = (char  *)inGmac_.get();
    if(ret == NULL) return 0;
    else if(*ret == gmacCode) return 1;
    return 0;
}

