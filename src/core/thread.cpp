#include "core/thread.h"

namespace __impl { namespace core {

__impl::util::Private<thread> TLS::CurrentThread_;

#ifdef DEBUG
Atomic thread::NextTID_ = 0;
#endif

}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
