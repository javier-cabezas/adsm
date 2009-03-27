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

#ifndef __PARAVER_TYPES_H_
#define __PARAVER_TYPES_H_

#include <paraver/Names.h>

namespace paraver {

STATE(_None_, 0x00);
STATE(_Running_, 0x01);
STATE(_Waiting_, 0x02);
STATE(_Create_, 0x03);
STATE(_IORead_, 0x04);
STATE(_IOWrite_, 0x05);

EVENT(_Alarm_, 0x00);

STATE(_gmacMalloc_, 0x10);
STATE(_gmacFree_, 0x11);
STATE(_gmacLaunch_, 0x13);
STATE(_gmacSync_, 0x14);

STATE(_accMalloc_, 0x20);
STATE(_accFree_, 0x21);
STATE(_accMemcpy_, 0x22);
STATE(_accLaunch_, 0x23);
STATE(_accSync_, 0x24);

EVENT(_gpuMemcpyToDevice_, 0x20);
EVENT(_gpuMemcpyToHost_, 0x21);

EVENT(_gpuLaunch_, 0x30);

STATE(_gmacSignal_, 0x40);

};


#endif
