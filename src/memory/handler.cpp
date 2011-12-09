#include "handler.h"

namespace __impl { namespace memory {
handler *handler::Handler_ = NULL;

handler::CallBack handler::Entry_ = NULL;
handler::CallBack handler::Exit_ = NULL;

} };
