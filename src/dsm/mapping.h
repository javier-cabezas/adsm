/* Copyright (c) 2009-2011 University of Illinois
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

#ifndef GMAC_DSM_MAPPING_H_
#define GMAC_DSM_MAPPING_H_

#include "hal/types.h"

#include "util/factory.h"
#include "util/misc.h"

#include "coherence.h"

namespace __impl { namespace dsm {

class GMAC_LOCAL mapping :
    public util::factory<coherence::block,
                         coherence::block_ptr> {

protected:
    hal::ptr addr_;
    size_t size_;

    typedef util::factory<coherence::block,
                          coherence::block_ptr> factory_block;

    typedef std::list<coherence::block_ptr> list_block;
    list_block blocks_;

    typedef util::range<list_block::iterator> range_block;
    range_block get_blocks_in_range(size_t offset, size_t count);

    template <typename I>
    static mapping_ptr merge_mappings(util::range<I> range, size_t off, size_t count);

    static gmacError_t dup2(mapping_ptr map1, hal::ptr::offset_type off1,
                            mapping_ptr map2, hal::ptr::offset_type off2, size_t count);

    gmacError_t dup(hal::ptr::offset_type off1, mapping_ptr map2,
                    hal::ptr::offset_type off2, size_t count);

    gmacError_t split(hal::ptr::offset_type off, size_t count);

    gmacError_t prepend(coherence::block_ptr b);
    gmacError_t append(coherence::block_ptr b);
    gmacError_t append(mapping_ptr map);

    mapping(hal::ptr addr);

public:
    static const size_t MinAlignment = 4096;

    typedef std::list<mapping_ptr> submappings;

    gmacError_t acquire(size_t offset, size_t count, int flags);
    gmacError_t release(size_t offset, size_t count);

    template <typename I>
    static gmacError_t link(hal::ptr ptr1, util::range<I> range1, submappings &sub1,
                            hal::ptr ptr2, util::range<I> range2, submappings &sub2, size_t count, int flags);

    //static submappings split(hal::ptr addr, size_t count);

    typedef util::bounds<size_t> bounds;
    bounds get_bounds() const;

    hal::ptr get_ptr() const;
};

}}

#include "mapping-impl.h"

#endif // GMAC_DSM_MAPPING_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
