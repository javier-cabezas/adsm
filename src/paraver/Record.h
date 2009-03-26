/* Copyright (c) 2009 University of Illinois
                   Universitat Politecnica de Catalunya
                   All rights reserved.

Developed by: IMPACT Research Group / Grup de Sistemes Operatius
              University of Illinois / Universitat Politecnica de Catalunya
              http://impact.crhc.illinois.edu/
              http://gso.ac.upc.edu/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal with the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
  1. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimers.
  2. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimers in the
     documentation and/or other materials provided with the distribution.
  3. Neither the names of IMPACT Research Group, Grup de Sistemes Operatius,
     University of Illinois, Universitat Politecnica de Catalunya, nor the
     names of its contributors may be used to endorse or promote products
     derived from this Software without specific prior written permission.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
WITH THE SOFTWARE.  */

#ifndef __PARAVER_RECORD_H_
#define __PARAVER_RECORD_H_

#include <config/config.h>
#include <paraver/Time.h>

#include <assert.h>

#include <list>
#include <string>
#include <iostream>
#include <fstream>

namespace paraver {

class Thread;

class Record {
protected:
	typedef enum {
		STATE=1,
		EVENT=2,
		COMM=3,
		LAST
	} Type;
public:
	virtual Time_t getTime() const = 0;
	virtual Time_t getEndTime() const = 0;
	virtual int getType() const = 0;
	virtual void write(std::ofstream &of) const = 0;

	static void end(std::ofstream &of) {
		Type type = LAST;
		of.write((char *)&type, sizeof(type));
	}
	static Record *read(std::ifstream &in);
	friend std::ostream & operator<<(std::ostream &os, const Record &record);
};

class RecordPredicate {
public:
	inline bool operator()(const Record *a, const Record *b) {
		if(a->getTime() < b->getTime()) return true;
		if(a->getTime() == b->getTime()) {
			a->getType() > b->getType();	
		}
		return false;
	}
};

class RecordId {
protected:
	int32_t task, app, thread;
public:
	RecordId(int32_t task, int32_t app, int32_t thread) :
		task(task), app(app), thread(thread) {};

	RecordId(std::ifstream &in) {
		in.read((char *)&task, sizeof(task));
		in.read((char *)&app, sizeof(app));
		in.read((char *)&thread, sizeof(thread));
	}
	
	void write(std::ofstream &of) const {
		of.write((char *)&task, sizeof(task));
		of.write((char *)&app, sizeof(app));
		of.write((char *)&thread, sizeof(thread));
	}

	friend std::ostream & operator<<(std::ostream &os, const RecordId &id) {
		os << 0 << ":" << id.task << ":" << id.app << ":" << id.thread;
		return os;
	}
};

class State : public Record {
public:
	static const uint32_t None = 0;
	static const uint32_t Running = 1;
private:
	RecordId id;
	Time_t _start;
	Time_t _end;
	uint32_t _state;
public:
	State(Thread *thread);
	State(std::ifstream &in) : id(in) {
		in.read((char *)&_start, sizeof(_start));
		in.read((char *)&_end, sizeof(_end));
		in.read((char *)&_state, sizeof(_state));
	}

	inline int getType() const { return STATE; }
	inline Time_t getTime() const { return _start; }
	inline Time_t getEndTime() const { return _end; }

	inline void start(uint32_t state, Time_t start) { 
		_state = state;
		_start = start;
	}
	inline void restart(Time_t start) { _start = start; }
	inline void end(Time_t end) { _end = end; }

	void write(std::ofstream &of) const {
		if(_start == _end) return;
		assert(_start < _end);
		Type type = STATE;
		of.write((char *)&type, sizeof(type));
		id.write(of);
		of.write((char *)&_start, sizeof(_start));
		of.write((char *)&_end, sizeof(_end));
		of.write((char *)&_state, sizeof(_state));
	}
	friend std::ostream & operator<<(std::ostream &os, const State &state) {
		os << STATE << ":" << state.id << ":" << state._start << ":";
		os << state._end << ":" << state._state << std::endl;
		return os;
	}
};

class Event : public Record {
private:
	RecordId id;
	Time_t _when;
	uint32_t _event;
	uint32_t _value;
public:
	Event(Thread *thread, Time_t when, uint32_t event, uint32_t value);
	Event(std::ifstream &in) : id(in) {
		in.read((char *)&_when, sizeof(_when));
		in.read((char *)&_event, sizeof(_event));
		in.read((char *)&_value, sizeof(_value));
	}

	inline int getType() const { return EVENT; }
	inline Time_t getTime() const { return _when; }
	inline Time_t getEndTime() const { return _when; }

	void write(std::ofstream &of) const {
		Type type = EVENT;
		of.write((char *)&type, sizeof(type));
		id.write(of);
		of.write((char *)&_when, sizeof(_when));
		of.write((char *)&_event, sizeof(_event));
		of.write((char *)&_value, sizeof(_value));
	}
	friend std::ostream & operator<<(std::ostream &os, const Event &event) {
		os << EVENT << ":" << event.id << ":" << event._when << ":";
		os << event._event << ":" << event._value << std::endl;
		return os;
	}

};

};
#endif
