#include "Record.h"
#include "Element.h"

#include <common/debug.h>

namespace paraver {

Record *Record::read(std::ifstream &in)
{
	Type type;
	in.read((char *)&type, sizeof(type));
	assert(in.eof() == false);
	switch(type) {
		case STATE:
			return new State(in);
		case EVENT:
			return new Event(in);
		case LAST:
			return NULL;
		default:
			abort();
	};
}

std::ostream & operator<<(std::ostream &os, const Record &record)
{
	if(typeid(record) == typeid(State)) 
		os << dynamic_cast<const State &>(record);
	else if(typeid(record) == typeid(Event)) 
		os << dynamic_cast<const Event &>(record);
	
	return os;
}

State::State(Thread *thread) :
	id(thread->getTask(), thread->getApp(), thread->getId()),
	_start(0),
	_end(0),
	_state(-1)
{ }


Event::Event(Thread *thread, Time_t when, uint32_t event, uint32_t value) :
	id(thread->getTask(), thread->getApp(), thread->getId()),
	_when(when),
	_event(event),
	_value(value)
{}


};
