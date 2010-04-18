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

#ifndef __API_CUDADRV_MODULE_H_
#define __API_CUDADRV_MODULE_H_

#include <config.h>
#include <debug.h>

#include <list>
#include <vector>

#include "Kernel.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

namespace gmac { namespace gpu {

typedef const char * gmacVariable_t;
typedef const struct textureReference * gmacTexture_t;

typedef Descriptor<gmacTexture_t> TextureDescriptor;

class VariableDescriptor : public Descriptor<gmacVariable_t> {
protected:
    bool _constant;

public:
    VariableDescriptor(const char *name, gmacVariable_t key, bool constant);
    bool constant() const;
};

class Variable : public VariableDescriptor {
	CUdeviceptr _ptr;
    size_t _size;
public:
	Variable(const VariableDescriptor & v, CUmodule mod);
    size_t size() const;
    CUdeviceptr devPtr() const;
};

class Texture : public TextureDescriptor {
protected:
    CUtexref _texRef;

public:
	Texture(const TextureDescriptor & t, CUmodule mod);

    CUtexref texRef() const;
};

class Module;
typedef std::vector<Module *> ModuleVector;

class ModuleDescriptor {
    static const char *pageTableSymbol;
    typedef std::vector<ModuleDescriptor *> ModuleDescriptorVector;
    static ModuleDescriptorVector Modules;
	const void * _fatBin;

    typedef std::vector<gmac::KernelDescriptor> KernelVector;
    typedef std::vector<VariableDescriptor>     VariableVector;
	typedef std::vector<TextureDescriptor>      TextureVector;

    KernelVector   _kernels;
	VariableVector _variables;
	VariableVector _constants;
	TextureVector  _textures;

    friend class Module;

    VariableDescriptor * _pageTable;

public:
    ModuleDescriptor(const void * fatBin);

    void add(gmac::KernelDescriptor & k);
    void add(VariableDescriptor     & v);
    void add(TextureDescriptor      & t);

    const VariableDescriptor & pageTable() const;
    static ModuleVector createModules();
};

class Module {
protected:
	CUmodule _mod;
	const void *_fatBin;

	typedef std::map<gmacVariable_t, Variable> VariableMap;
	typedef std::map<gmacTexture_t, Texture> TextureMap;

    VariableMap _variables;
	VariableMap _constants;
	TextureMap  _textures;

	Variable *_pageTable;

public:
	Module(const ModuleDescriptor & d);
	~Module();

    const Variable *variable(gmacVariable_t key) const;
	const Variable *constant(gmacVariable_t key) const;
    const Texture  *texture(gmacTexture_t   key) const;
};

}}

#include "Module.ipp"

#endif
