#include "Accelerator.h"

namespace gmac {

Accelerator::Accelerator(int n) :
    logger("Accelerator"), _memory(0), _id(n)
{
}

Accelerator::~Accelerator()
{
}

}
