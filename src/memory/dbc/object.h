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

#ifndef GMAC_MEMORY_DBC_OBJECT_H_
#define GMAC_MEMORY_DBC_OBJECT_H_

namespace __dbc { namespace memory {

class GMAC_LOCAL object :
    public __impl::memory::object,
    public virtual Contract {
    DBC_TESTED(__impl::memory::object)

protected:
    typedef __impl::memory::object parent;

    typedef __impl::memory::protocol protocol_impl;
    typedef __impl::hal::event_ptr event_ptr_impl;
    typedef __impl::hal::device_output device_output_impl;
    typedef __impl::hal::device_input device_input_impl;

	object(protocol_impl &protocol, host_ptr addr, size_t size);
    virtual ~object();

public:
    ssize_t get_block_base(size_t offset) const;
    size_t get_block_end(size_t offset) const;

    event_ptr_impl signal_read(host_ptr addr, gmacError_t &err);
    event_ptr_impl signal_write(host_ptr addr, gmacError_t &err);

    gmacError_t to_io_device(device_output_impl &output, size_t offset, size_t count);
    gmacError_t from_io_device(size_t offset, device_input_impl &input, size_t count);

    gmacError_t memset(size_t offset, int v, size_t size);

    gmacError_t memcpy_to_object(size_t objOffset,
                                 host_const_ptr src, size_t count);

    gmacError_t memcpy_object_to_object(object &dstObj, size_t dstOffset,
                                        size_t srcOffset,
                                        size_t count);

    gmacError_t memcpy_from_object(host_ptr dst,
                                   size_t objOffset, size_t count);
};

}}

#endif /* OBJECT_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
