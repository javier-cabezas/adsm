#include "Record.h"
#include "Element.h"

#include <typeinfo>

namespace __impl { namespace trace { namespace paraver {

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
	return NULL;
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
	id_(thread->getTask(), thread->getApp(), thread->getId()),
	start_(0),
	end_(0),
	state_(-1)
{ }


Event::Event(Thread *thread, uint64_t when, uint64_t event, int64_t value) :
	id_(thread->getTask(), thread->getApp(), thread->getId()),
	when_(when),
	event_(event),
	value_(value)
{}


} } }
