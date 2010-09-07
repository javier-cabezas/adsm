#include "Function.h"
#include "Logger.h"
#include <cassert>

#ifdef PARAVER
namespace gmac { namespace util {
const char *Function::eventName = "Function";
Function::FunctionMap *Function::map = NULL;
paraver::EventName *Function::event = NULL;
#endif

void Function::start(const char *name)
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    if(event == NULL)
        event = paraver::Factory<paraver::EventName>::create(eventName);
    if(map == NULL) map = new FunctionMap();
    FunctionMap::const_iterator i = map->find(std::string(name));

    unsigned id = -1;
    if(i == map->end()) {
        id = map->size() + 1;
        event->registerType(id, std::string(name));
        map->insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;

    paraver::trace->__pushEvent(*event, id);
#endif
}

void Function::end()
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    Logger::ASSERTION(event != NULL);
    Logger::ASSERTION(map != NULL);

    paraver::trace->__pushEvent(*event, 0);
#endif
}

}}
