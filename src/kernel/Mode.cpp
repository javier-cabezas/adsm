#include "Mode.h"
#include "Accelerator.h"

#include <memory/Object.h>
#include <memory/Map.h>

namespace gmac {

gmac::util::Private Mode::key;

Mode::Mode(Accelerator *acc) :
    __acc(acc)
{
    __context = __acc->create();
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

void *Mode::translate(void *addr)
{
    memory::Object *object = __map->find(addr);
    if(object == NULL) object = proc->shared().find(addr)->second;
    if(object == NULL) return NULL;
    off_t offset = (uint8_t *)addr - (uint8_t *)object->addr();
    uint8_t *ret= (uint8_t *)object->device() + offset;
    return ret; 
}

}
