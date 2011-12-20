/* Copyright (c) 2011 University of Illinois
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

#ifndef GMAC_MEMORY_PROTOCOL_COMMON_BLOCK_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_COMMON_BLOCK_IMPL_H_

namespace __impl {
namespace memory { namespace protocols { namespace common {

inline
block::block(object &parent, hostptr_t addr, hostptr_t shadow, size_t size) :
    Lock("block"),
    parent_(parent),
    size_(size),
    addr_(addr),
    shadow_(shadow),
    faultsCacheWrite_(0),
    faultsCacheRead_(0)
{
}

inline const block::bounds
block::get_bounds() const
{
    // No need for lock -- addr_ is never modified
    return bounds(addr_, addr_ + size_);
}

inline size_t
block::size() const
{
    return size_;
}

inline hostptr_t
block::get_shadow() const
{
    return shadow_;
}

inline
unsigned
block::get_faults_cache_write() const
{
    return faultsCacheWrite_;
}

inline
unsigned
block::get_faults_cache_read() const
{
    return faultsCacheRead_;
}

inline
void
block::reset_faults_cache_write()
{
    faultsCacheWrite_ = 0;
}

inline
void
block::reset_faults_cache_read()
{
    faultsCacheRead_ = 0;
}

}}}}

#endif /* BLOCKSTATE_IMPL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
