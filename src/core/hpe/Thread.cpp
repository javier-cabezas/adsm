#include "core/hpe/Thread.h"

namespace __impl { namespace core { namespace hpe {

PRIVATE Mode *TLS::CurrentMode_ = NULL;
PRIVATE Thread *TLS::CurrentThread_ = NULL;

}}}
/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
