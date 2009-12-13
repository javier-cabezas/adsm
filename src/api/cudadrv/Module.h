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

#ifndef __API_CUDADRV_GPUMODULE_H_
#define __API_CUDADRV_GPUMODULE_H_

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

	Function(CUmodule mod, const char *name) :
			dev(dev), name(name) {
		load(mod);
	}

	inline void load(CUmodule mod) {
		assert(cuModuleGetFunction(&fun, mod, name) == CUDA_SUCCESS);
	} 
};

class Variable {
public:
	const char *dev;
	CUdeviceptr ptr;
	size_t size;

	Variable(const char *dev, CUdeviceptr ptr, size_t size) :
		dev(dev), ptr(ptr), size(size) {};
};

class Texture {
public:
	const char *name;
	struct __textureReference *ref;
	Texture(CUmodule mod, struct __textureReference *ref,
			const char *name) : name(name), ref(ref) {
		load(mod);
	}

	inline void load(CUmodule mod) {
		assert(cuModuleGetTexRef(&ref->__texref, mod, name) == CUDA_SUCCESS);
	}
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

	inline void reload() {
		TRACE("Module image: %p", fatBin);
		CUresult r = cuModuleLoadFatBinary(&mod, fatBin);
		assert(r == CUDA_SUCCESS);
		FunctionMap::iterator f;
		for(f = functions.begin(); f != functions.end(); f++)
			f->second.load(mod);
		TextureList::iterator t;
		for(t = textures.begin(); t != textures.end(); t++)
			t->load(mod);
	}


public:
	Module(const void *fatBin) : fatBin(fatBin), _pageTable(NULL) {
		TRACE("Module image: %p", fatBin);
		assert(cuModuleLoadFatBinary(&mod, fatBin) == CUDA_SUCCESS);
	}

	Module(const Module &root) :
		fatBin(root.fatBin), functions(root.functions),
		variables(root.variables), constants(root.constants),
		textures(root.textures)
	{
		reload();
	}

	~Module() {
 		assert(cuModuleUnload(mod) == CUDA_SUCCESS);
	}


	inline void function(const char *host, const char *dev) {
		functions.insert(FunctionMap::value_type(host, Function(mod, dev)));
	}

	inline const Function *function(const char *name) const {
		FunctionMap::const_iterator f;
		f = functions.find(name);
		if(f == functions.end()) return NULL;
		return &f->second;
	}

	inline void variable(const char *host, const char *dev) {
		CUdeviceptr ptr; 
		unsigned int size;
		CUresult ret = cuModuleGetGlobal(&ptr, &size, mod, dev);
		variables.insert(VariableMap::value_type(host, Variable(dev, ptr, size)));
	}

	inline const Variable *variable(const char *name) const {
		VariableMap::const_iterator v;
		v = variables.find(name);
		if(v == variables.end()) return NULL;
		return &v->second;
	}

	inline void constant(const char *host, const char *dev) {
		CUdeviceptr ptr; 
		unsigned int size;
		std::pair<VariableMap::iterator, bool> var;
		CUresult ret = cuModuleGetGlobal(&ptr, &size, mod, dev);
		var = constants.insert(VariableMap::value_type(host,
				Variable(dev, ptr, size)));
		if(strncmp(dev, pageTableSymbol, strlen(pageTableSymbol)) == 0)
			_pageTable = &var.first->second;
	}

	inline const Variable *constant(const char *name) const {
		VariableMap::const_iterator v;
		v = constants.find(name);
		if(v == constants.end()) return NULL;
		return &v->second;
	}

	inline Variable *pageTable() const { return _pageTable; }

	inline void texture(struct __textureReference *ref, const char *name) {
		textures.push_back(Texture(mod, ref, name));
	}
};

}}

#endif
