#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/ContextFactory.h"

namespace __impl { namespace opencl { namespace hpe {

Context *ContextFactory::create(Mode &mode, cl_command_queue stream) const
{
    return new Context(mode, stream);
}

#if 0
void ContextFactory::destroy(Context &context) const
{
    delete &context;
}
#endif

}}}
