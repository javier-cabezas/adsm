#include "config/order.h"

#include "libs/common.h"

#include "util/Atomics.h"
#include "util/Private.h"

static __impl::util::Private<const char> inGmac_;

static const char gmacCode = 1;
static const char userCode = 0;

static Atomic gmacInit__ = 0;

static volatile bool gmacIsInitialized = false;

CONSTRUCTOR(init);
static void init(void)
{
    /* Create GMAC enter lock and set GMAC as initialized */
    __impl::util::Private<const char>::init(inGmac_);
#ifdef POSIX
    threadInit();
#endif
}

void enterGmac()
{
    if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) {
        inGmac_.set(&gmacCode);
        initGmac();
        gmacIsInitialized = true;
        inGmac_.set(&userCode);
    } else {
        while (!gmacIsInitialized);
    }
    inGmac_.set(&gmacCode);
}


void enterGmacExclusive()
{
    if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) initGmac();
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
