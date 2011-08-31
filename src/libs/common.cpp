#include "config/order.h"

#include "libs/common.h"

#include "util/Atomics.h"
#include "util/Private.h"

PRIVATE static bool inGmac_   = false;
PRIVATE bool isRunTimeThread_ = false;

static const char gmacCode = 1;
static const char userCode = 0;

static Atomic gmacInit__ = 0;

static volatile bool gmacIsInitialized = false;

CONSTRUCTOR(init);
static void init(void)
{
#ifdef POSIX
    threadInit();
#endif
}

void enterGmac()
{
    if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) {
        inGmac_ = true;
        initGmac();
        gmacIsInitialized = true;
    } else if (isRunTimeThread_ == false) {
        while (!gmacIsInitialized);
        inGmac_ = true;
    }
}

void enterGmacExclusive()
{
    if(AtomicTestAndSet(gmacInit__, 0, 1) == 0) initGmac();
    inGmac_ = true;
}

void exitGmac()
{
    inGmac_ = false;
}

bool inGmac()
{
    return inGmac_;
}
