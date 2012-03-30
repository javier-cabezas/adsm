/* Copyright (c) 2009-2012 University of Illinois
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

#ifndef GMAC_HAL_CUDA_MODULE_H_
#define GMAC_HAL_CUDA_MODULE_H_

#include <cuda.h>
#include <driver_types.h>
#include <cuda_texture_types.h>

#include <list>
#include <map>
#include <string>
#include <vector>

#include "config/common.h"
#include "config/config.h"

#include "util/descriptor.h"
#include "util/smart_ptr.h"
#include "util/stl/locked_map.h"

#include "types.h"

namespace __impl { namespace hal { namespace cuda { namespace code {

typedef const char *cuda_variable_t;
typedef const struct textureReference *cuda_texture_t;

typedef util::descriptor<gmac_kernel_id_t> kernel_descriptor;
typedef util::descriptor<cuda_texture_t> texture_descriptor;

class repository;

typedef util::stl::locked_map<virt::aspace *, repository *> map_context_repository;


class GMAC_LOCAL variable_descriptor : public util::descriptor<cuda_variable_t> {
protected:
    bool constant_;

public:
    variable_descriptor(const std::string &name, cuda_variable_t key, bool constant);
    bool constant() const;
};

class GMAC_LOCAL variable_t : public variable_descriptor {
	CUdeviceptr ptr_;
    size_t size_;
public:
	variable_t(const variable_descriptor & v, CUmodule mod);
    size_t size() const;
    CUdeviceptr devPtr() const;
};

class GMAC_LOCAL texture_t : public texture_descriptor {
protected:
    CUtexref texRef_;

public:
	texture_t(const texture_descriptor & t, CUmodule mod);

    CUtexref texRef() const;
};

class GMAC_LOCAL module_descriptor {
	friend class repository;

protected:
	const void *fatBin_;

    typedef std::vector<kernel_descriptor>   vector_kernel;
    typedef std::vector<variable_descriptor> vector_variable;
	typedef std::vector<texture_descriptor>  vector_texture;

    vector_kernel   kernels_;
	vector_variable variables_;
	vector_variable constants_;
	vector_texture  textures_;

public:
    module_descriptor(const void *fatBin);

    void add(kernel_descriptor   &k);
    void add(variable_descriptor &v);
    void add(texture_descriptor  &t);

#if 0
    static repository *create_modules();
#endif
};

typedef std::vector<module_descriptor *> vector_module_descriptor;

class GMAC_LOCAL repository :
    public hal::detail::code::repository_mapping {
private:
    vector_module_descriptor descriptors_;

public:
    gmacError_t 
    load_from_file(const std::string &path,
                   const std::string &flags);

    gmacError_t
    load_from_mem(const char *ptr,
                  size_t size,
                  const std::string &flags);

    gmacError_t
    load_from_handle(const char *ptr,
                     const std::string &flags);
};


typedef hal::detail::code::kernel hal_kernel;

class GMAC_LOCAL repository_mapping :
    public hal::detail::code::repository {
protected:
	std::vector<CUmodule> mods_;
	const void *fatBin_;

	typedef std::map<cuda_variable_t, variable_t> map_variable;
	typedef std::map<std::string, variable_t> map_variable_name;
	typedef std::map<cuda_texture_t, texture_t> map_texture;
    typedef std::map<gmac_kernel_id_t, kernel *> map_kernel;

    map_kernel kernels_;
    map_variable variables_;
	map_variable constants_;
	map_texture textures_;

    map_variable_name kernelsByName_;
    map_variable_name variablesByName_;
	map_variable_name constantsByName_;

public:
	repository(virt::aspace &as);
	repository(virt::aspace &as, const repository &repo);

	~repository();

#if 0
    template <typename T>
    void register_kernels(T &t) const;
#endif

    hal_kernel *get_kernel(gmac_kernel_id_t key);
    hal_kernel *get_kernel(const std::string &name);

    const variable_t *variable(cuda_variable_t key) const;
	const variable_t *constant(cuda_variable_t key) const;
    const variable_t *variableByName(const std::string &name) const;
	const variable_t *constantByName(const std::string &name) const;
    const texture_t  *texture(cuda_texture_t key) const;
};

}}}}

#include "module-impl.h"

#endif
