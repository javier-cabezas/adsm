#include "init.h"

#include "util/Private.h"
#include "util/Atomics.h"

namespace __impl {
class GMACLock {
public:
    GMACLock() {}

    void lockRead()  const { }
    void lockWrite() const { }
    void unlock()    const { }
};

util::Private<const char> _inGmac;
GMACLock * _inGmacLock;
const char _gmacCode = 1;
const char _userCode = 0;

Atomic gmacInit__ = 0;
Atomic gmacFini__ = 0;

void init() { }

}

static bool Trace_ = false;

void InitGmac()
{
    __impl::util::Private<const char>::init(__impl::_inGmac);
    __impl::_inGmacLock = new __impl::GMACLock();
    __impl::_inGmac.set(&__impl::_gmacCode);
    __impl::gmacInit__ = 1;
    __impl::_inGmac.set(&__impl::_userCode);
}

void InitTrace(void)
{
    if(Trace_ == true) return;
    Trace_ = true;
}

void FiniTrace(void)
{
    if(Trace_ == false) return;
    Trace_ = false;
}

namespace __impl {

void enterGmac() { }
void exitGmac() { }

}
