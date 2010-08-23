#include "Mode.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>

namespace gmac {

gmac::util::Private Mode::key;

Mode::Mode(Accelerator *acc) :
    __acc(acc)
{
    trace("Creating new context");
    __context = __acc->create();
    trace("Creating new memory map");
    __map = new memory::Map(paraver::LockMmLocal);
}

Mode::~Mode()
{
    __count--;
    if(__count > 0)
        gmac::util::Logger::WARNING("Deleting in-use Execution Mode");
    if(this == key.get()) key.set(NULL);
    delete __map;
    __acc->destroy(__context); 
}

void Mode::switchTo(Accelerator *acc)
{
}


}
