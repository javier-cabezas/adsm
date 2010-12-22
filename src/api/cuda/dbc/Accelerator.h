/*
 * <+ DESCRIPTION +>
 *
 * Copyright (C) 2010, Javier Cabezas <jcabezas in ac upc edu> {{{
 *
 * This program is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU General Public License 
 * as published by the Free Software Foundation; either 
 * version 2 of the License, or any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 * }}}
 */

#ifndef GMAC_API_CUDA_DBC_ACCELERATOR_H_
#define GMAC_API_CUDA_DBC_ACCELERATOR_H_

namespace __dbc { namespace cuda {

class GMAC_LOCAL Accelerator :
    public __impl::cuda::Accelerator,
    public virtual Contract {
    DBC_TESTED(__impl::cuda::Accelerator)

public:
	Accelerator(int n, CUdevice device);
    virtual ~Accelerator();

    /* Synchronous interface */
	gmacError_t copyToAccelerator(accptr_t acc, const hostptr_t host, size_t size);
	gmacError_t copyToHost(hostptr_t host, const accptr_t acc, size_t size);
	gmacError_t copyAccelerator(accptr_t dst, const accptr_t src, size_t size);

    /* Asynchronous interface */
    gmacError_t copyToAcceleratorAsync(accptr_t acc, __impl::cuda::IOBuffer &buffer, size_t bufferOff, size_t count, __impl::cuda::Mode &mode, CUstream stream);
    gmacError_t copyToHostAsync(__impl::cuda::IOBuffer &buffer, size_t bufferOff, const accptr_t acc, size_t count, __impl::cuda::Mode &mode, CUstream stream);
};


}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
