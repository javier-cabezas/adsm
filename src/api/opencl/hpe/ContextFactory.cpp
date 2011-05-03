#include "api/opencl/hpe/Context.h"
#include "api/opencl/hpe/ContextFactory.h"

namespace __impl { namespace opencl { namespace hpe {

Context *ContextFactory::create(Mode &mode) const
{
    return new Context(mode);
}

void ContextFactory::destroy(Context &context) const
{
    delete &context;
}

}}}
