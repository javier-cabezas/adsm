#include <cassert>
#include <list>
#include <ostream>

#include "util/Logger.h"
#include "Function.h"

namespace gmac { namespace trace {

#ifdef DEBUG
util::Private<std::list<std::string> > Function::Funcs_;
#endif

#ifdef PARAVER
ModuleMap *Function::map = NULL;

FunctionMap::FunctionMap(unsigned n, const char *name)
{
    _event = paraver::Factory<paraver::EventName>::create(name);
    id_ = n * _stride;
}


FunctionMap &ModuleMap::get(const char *module)
{
    FunctionMap *func = NULL;
    iterator i = find(std::string(module));
    if(i != end())  func = i->second;
    else {
        func = new FunctionMap(size(), module);
        insert(value_type(std::string(module), func));
    }
    return *func;
}

#endif



void Function::init()
{
#ifdef DEBUG
    util::Private<std::list<std::string> >::init(Funcs_);
    Funcs_.set(NULL);
#endif
#ifdef PARAVER
    map = new ModuleMap();
#endif
}

void Function::initThread()
{
#ifdef DEBUG
    Funcs_.set(new std::list<std::string>());
#endif
}

void Function::start(const char *module, const char *name)
{
#ifdef DEBUG
    gmac::util::Logger::TRACE("FUNCTION START %s:%s", module, name);
    Funcs_.get()->push_back(std::string(name));
#endif
#ifdef PARAVER
    if(paraver::trace == NULL) return;
    map->lock();
    FunctionMap &function = map->get(module);

    FunctionMap::const_iterator i = function.find(std::string(name));
    unsigned id = -1;
    if(i == function.end()) {
        id = function.id() + function.size() + 1;
        function.event().registerType(id, std::string(name));
        function.insert(FunctionMap::value_type(std::string(name), id));
    }
    else id = i->second;
    map->unlock();
    paraver::trace->__pushEvent(function.event(), id);
#endif
}

void Function::end(const char *module)
{
#ifdef PARAVER
    if(paraver::trace == NULL) return;

    FunctionMap &function = map->get(module);
    paraver::trace->__pushEvent(function.event(), 0);
#endif
#ifdef DEBUG
    gmac::util::Logger::ASSERTION(Funcs_.get()->size() > 0);
    gmac::util::Logger::TRACE("FUNCTION END %s:%s", module, Funcs_.get()->back().c_str());
    Funcs_.get()->pop_back();
#endif
}

}}
