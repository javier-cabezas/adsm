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

#include "api/cuda/Accelerator.h"
#include "api/cuda/Mode.h"

#include "memory/Bitmap.h"

#ifdef USE_VM
namespace gmac { namespace memory { namespace vm {

void Bitmap::allocate()
{
    assertion(_device == NULL);
    gmac::cuda::Mode * mode = gmac::cuda::Mode::current();
#ifdef USE_HOSTMAP_VM
    mode->hostAlloc((void **)&_bitmap, _size);
    _device = mode->hostMap(_bitmap);
    memset(_bitmap, 0, size());
    trace("Allocating dirty bitmap (%zu bytes)", size());
#else
    mode->malloc((void **)&_device, _size);
    trace("Allocating dirty bitmap %p -> %p (%zu bytes)", _bitmap, _device, _size);
#endif
}

Bitmap::~Bitmap()
{
    gmac::cuda::Mode * mode = gmac::cuda::Mode::current();
    if(_device != NULL) mode->hostFree(_bitmap);
}

void
Bitmap::sync()
{
#ifndef USE_HOSTMAP_VM
    trace("Syncing Bitmap");
    Mode * mode = Mode::current();
    gmac::memory::vm::Bitmap & bitmap = mode->dirtyBitmap();
    trace("Setting dirty bitmap on host: %p -> %p: %zd", (void *) cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.host(), bitmap.size());
    const void *__device = bitmap.deviceBase();
    CUresult ret;
    //printf("Bitmap toHost\n");
    ret = cuMemcpyDtoH(bitmap.host(), cuda::Accelerator::gpuAddr(bitmap.device()), bitmap.size());
    cfatal(ret == CUDA_SUCCESS, "Unable to copy back dirty bitmap");
    ret = cuCtxSynchronize();
    cfatal(ret == CUDA_SUCCESS, "Unable to wait for copy back dirty bitmap");
    bitmap.reset();

    _synced = true;
#endif
}

}}}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
