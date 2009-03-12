#include "Record.h"
#include "Element.h"

namespace paraver {

Record *Record::read(std::ifstream &in)
{
	Type type;
	in.read((char *)&type, sizeof(type));
	switch(type) {
		case STATE:
			return new State(in);
		case EVENT:
			return new Event(in);
		case COMM:
		case LAST:
			return NULL;
	};
}


State::State(Thread *thread) :
	id(thread->getCpu(), thread->getTask(), thread->getApp(), thread->getId()),
	_start(0),
	_end(0),
	_state(-1)
{}


Event::Event(Thread *thread, Time_t when, uint32_t event, uint32_t value) :
	id(thread->getCpu(), thread->getTask(), thread->getApp(), thread->getId()),
	_when(when),
	_event(event),
	_value(value)
{}


};
