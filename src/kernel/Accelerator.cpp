#include "Accelerator.h"

namespace gmac {

Accelerator::Accelerator(int n) :
    _memory(0), _id(n), _load(0)
{
}

Accelerator::~Accelerator()
{
}

}
