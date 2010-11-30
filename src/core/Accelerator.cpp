#include "Accelerator.h"

namespace __impl { namespace core {

Accelerator::Accelerator(int n) :
    memory_(0), id_(n), load_(0)
{
}

Accelerator::~Accelerator()
{
}

}}
