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

Atomic _gmacInit = 0;

void init() { }

}

void InitGmac()
{
    __impl::util::Private<const char>::init(__impl::_inGmac);
    __impl::_inGmacLock = new __impl::GMACLock();
    __impl::_inGmac.set(&__impl::_gmacCode);
    __impl::_gmacInit = 1;
    __impl::_inGmac.set(&__impl::_userCode);
}
