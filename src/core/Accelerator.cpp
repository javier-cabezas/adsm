#include "trace/Tracer.h"
#include "util/Logger.h"

#include "Accelerator.h"

namespace __impl { namespace core {

Accelerator::Accelerator(int n) :
    id_(n), load_(0)
{
}

Accelerator::~Accelerator()
{
}

void Accelerator::registerMode(Mode &mode)
{
    TRACE(LOCAL,"Registering Execution Mode %p to Accelerator", &mode);
    trace::EnterCurrentFunction();
    queue_.insert(&mode);
    load_++;
    trace::ExitCurrentFunction();
}

void Accelerator::unregisterMode(Mode &mode)
{
    TRACE(LOCAL,"Unregistering Execution Mode %p", &mode);
    trace::EnterCurrentFunction();
    std::set<Mode *>::iterator c = queue_.find(&mode);
    ASSERTION(c != _queue.end());
    queue_.erase(c);
    load_--;
    trace::ExitCurrentFunction();
}

}}
