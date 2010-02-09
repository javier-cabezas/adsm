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

#ifndef __API_CUDADRV_MODULE_H_
#define __API_CUDADRV_MODULE_H_

#include <config.h>
#include <debug.h>

#include <cassert>

#include <list>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

struct __textureReference {
	int normalized;
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureAddressMode addressMode[3];
	struct cudaChannelFormatDesc channelDesc;
	CUtexref __texref;
	int __reserved[16 - sizeof(CUtexref)];
};


namespace gmac { namespace gpu {

class Function {
public:
	CUfunction fun;
	const char *dev;
	const char *name;

	Function(CUmodule mod, const char *name);

	void load(CUmodule mod);
};

class Variable {
public:
	const char *dev;
	CUdeviceptr ptr;
	size_t size;

	Variable(const char *dev, CUdeviceptr ptr, size_t size);
};

class Texture {
public:
	const char *name;
	struct __textureReference *ref;
	Texture(CUmodule mod, struct __textureReference *ref, const char *name);

	void load(CUmodule mod);
};

class Module {
protected:
	CUmodule mod;
	const void *fatBin;

	typedef HASH_MAP<const char *, Function> FunctionMap;
	typedef HASH_MAP<const char *, Variable> VariableMap;
	typedef std::list<Texture> TextureList;

	FunctionMap functions;
	VariableMap variables;
	VariableMap constants;
	TextureList textures;

	static const char *pageTableSymbol;
	Variable *_pageTable;

	void reload();

public:
	Module(const void *fatBin);
	Module(const Module &root);
	~Module();

	void function(const char *host, const char *dev);
	const Function *function(const char *name) const;

	void variable(const char *host, const char *dev);
	const Variable *variable(const char *name) const;

	void constant(const char *host, const char *dev);
	const Variable *constant(const char *name) const;

	Variable *pageTable() const;

	void texture(struct __textureReference *ref, const char *name);
};

#include "Module.ipp"

}}

#endif
