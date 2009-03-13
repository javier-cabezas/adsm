#include "Element.h"

namespace paraver {

void Thread::start(std::ofstream &of, unsigned s, Time_t t)
{ 
	assert(states.empty() == false);
	// Flush the previous state to disk (if needed)
	states.back().end(t);
	states.back().write(of);

	// Setup the new state
	states.push_back(State(this));
	states.back().start(s, t);
}

void Thread::end(std::ofstream &of, Time_t t)
{
	assert(states.empty() == false);
	// Flush previous state to disk (if needed)
	states.back().end(t);
	states.back().write(of);

	// Remove the old state
	states.pop_back();
	if(states.empty() == false) states.back().restart(t);
}


std::ostream &operator<<(std::ostream &os, const Application &app) 
{
	HASH_MAP<int32_t, Task *>::const_iterator i;
	os << app.sons.size();
	os << "(";
	for(i = app.sons.begin(); i != app.sons.end(); i++) {
		if(i != app.sons.begin()) os << ",";
		os << i->second->size() << ":" << 0;
	}
	os << ")";
	return os;
}

};
