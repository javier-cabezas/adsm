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

#ifndef __PARAVER_FUNCTION_H_
#define __PARAVER_FUNCTION_H_

#include <gmac/paraver.h>
#include <util/Lock.h>

#include <map>

namespace gmac { namespace trace {

class FunctionMap : public std::map<std::string, unsigned> {
protected:
    unsigned _id;
    static unsigned _count;
    static const unsigned stride = 256;
public:
    FunctionMap() { _id = _count; _count += stride; }
    inline unsigned id() const { return _id; }
};

class ModuleMap : public std::map<std::string, FunctionMap>, protected util::Lock {
protected:
    friend class Function;
public:
    ModuleMap() : Lock("Paraver") {};
};

class Function {
protected:
#ifdef PARAVER
    static const char *eventName;
    static ModuleMap *map;
    static paraver::EventName *event;

    static FunctionMap &getFunctionMap(const char *module);
#endif
public:
    static void init();
    static void start(const char *module, const char *name);
    static void end();
};

}}

#endif
