/* Copyright (c) 2009, 2010 University of Illinois
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

#ifndef GMAC_API_CUDA_MODULE_H_
#define GMAC_API_CUDA_MODULE_H_

#include "config/common.h"
#include "config/config.h"

#include "Kernel.h"

#include <list>
#include <vector>
#include <map>

#include <cuda.h>
#include <driver_types.h>
#include <texture_types.h>

namespace __impl { namespace cuda {

typedef const char * gmacVariable_t;
typedef const struct textureReference * gmacTexture_t;

typedef core::Descriptor<gmacTexture_t> TextureDescriptor;

class GMAC_LOCAL VariableDescriptor : public core::Descriptor<gmacVariable_t> {
protected:
    bool constant_;

public:
    VariableDescriptor(const char *name, gmacVariable_t key, bool constant);
    bool constant() const;
};

class GMAC_LOCAL Variable : public cuda::VariableDescriptor {
	CUdeviceptr ptr_;
    size_t size_;
public:
	Variable(const VariableDescriptor & v, CUmodule mod);
    size_t size() const;
    CUdeviceptr devPtr() const;
};

class GMAC_LOCAL Texture : public cuda::TextureDescriptor {
protected:
    CUtexref texRef_;

public:
	Texture(const TextureDescriptor & t, CUmodule mod);

    CUtexref texRef() const;
};

class Module;
typedef std::vector<Module *> ModuleVector;

class GMAC_LOCAL ModuleDescriptor {
	friend class Module;

protected:
    typedef std::vector<ModuleDescriptor *> ModuleDescriptorVector;
    static ModuleDescriptorVector Modules_;
	const void * fatBin_;

    typedef std::vector<core::KernelDescriptor> KernelVector;
    typedef std::vector<VariableDescriptor>     VariableVector;
	typedef std::vector<TextureDescriptor>      TextureVector;

    KernelVector   kernels_;
	VariableVector variables_;
	VariableVector constants_;
	TextureVector  textures_;

#ifdef USE_VM
    VariableDescriptor * dirtyBitmap_;
    VariableDescriptor * shiftPage_;
#ifdef BITMAP_BIT
    VariableDescriptor * shiftEntry_;
#endif
#endif

public:
    ModuleDescriptor(const void * fatBin);
    ~ModuleDescriptor();

    void add(core::KernelDescriptor & k);
    void add(VariableDescriptor     & v);
    void add(TextureDescriptor      & t);

    static ModuleVector createModules();


};

class GMAC_LOCAL Module {
protected:

	CUmodule mod_;
	const void *fatBin_;

	typedef std::map<gmacVariable_t, Variable> VariableMap;
	typedef std::map<gmacTexture_t, Texture> TextureMap;
    typedef std::map<const char *, Kernel *> KernelMap;

    VariableMap variables_;
	VariableMap constants_;
	TextureMap  textures_;
    KernelMap kernels_;

#ifdef USE_VM
    static const char *DirtyBitmapSymbol_;
    static const char *ShiftPageSymbol_;
	Variable *dirtyBitmap_;
	Variable *shiftPage_;
#ifdef BITMAP_BIT
    static const char *ShiftEntrySymbol_;
	Variable *shiftEntry_;
#endif
#endif

public:
	Module(const ModuleDescriptor & d);
	~Module();

    void registerKernels(Mode &mode) const;

    const Variable *variable(gmacVariable_t key) const;
	const Variable *constant(gmacVariable_t key) const;
    const Texture  *texture(gmacTexture_t   key) const;

#ifdef USE_VM
    const Variable *dirtyBitmap() const;
    const Variable *dirtyBitmapShiftPage() const;
#ifdef BITMAP_BIT
    const Variable *dirtyBitmapShiftEntry() const;
#endif
#endif
};

}}

#include "Module-impl.h"

#endif
