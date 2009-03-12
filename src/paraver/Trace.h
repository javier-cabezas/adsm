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

#ifndef __TRACE_H
#define __TRACE_H

#include "Element.h"
#include "Record.h"

#include <common/config.h>

#include <sys/time.h>

#include <assert.h>

#include <vector>
#include <list>
#include <string>
#include <iostream>
#include <fstream>

namespace paraver {

class Trace {
protected:
	std::vector<Node *> nodes;
	std::list<Application *> apps;

	inline static Time_t getTime() {
		struct timeval tv;
		gettimeofday(&tv, NULL);
		Time_t tm = tv.tv_usec + 1000000 * tv.tv_sec;
		return tm;
	}

	Time_t endTime;
	Time_t pendingTime;

	void setEnd(Time_t t) { endTime = (endTime > t) ? endTime : t; }
	void setPending(Time_t t) {
		pendingTime = (pendingTime > t) ? pendingTime : t;
	}

	void buildNode(std::ifstream &in);
	void buildApp(std::ifstream &in);

	std::ofstream of;
	std::list<Record *> records;

public:
	Trace(const char *fileName);

	void pushState(unsigned value);
	void popState();
	void event(unsigned type, unsigned value);

	void read(const char *filename);
	void write();
	friend std::ofstream &operator<<(std::ofstream &os, const Trace &trace);
};

};
#endif
