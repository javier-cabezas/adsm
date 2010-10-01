#include "Accelerator.h"

namespace gmac {

Accelerator::Accelerator(int n) :
    memory_(0), id_(n), load_(0)
{
}

Accelerator::~Accelerator()
{
}

}
