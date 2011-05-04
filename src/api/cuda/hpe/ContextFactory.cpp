#include "api/cuda/hpe/Context.h"
#include "api/cuda/hpe/ContextFactory.h"

namespace __impl { namespace cuda { namespace hpe {

Context *ContextFactory::create(Mode &mode) const
{
    return new Context(mode);
}

void ContextFactory::destroy(Context &context) const
{
    delete &context;
}

}}}
