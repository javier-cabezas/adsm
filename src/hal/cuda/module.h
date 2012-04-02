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


typedef hal::detail::code::repository hal_repository;

#if 0
typedef const char *cuda_variable_t;
typedef const struct textureReference *cuda_texture_t;

typedef util::descriptor<gmac_kernel_id_t> kernel_descriptor;
typedef util::descriptor<cuda_texture_t> texture_descriptor;


class GMAC_LOCAL variable_descriptor : public util::descriptor<cuda_variable_t> {
protected:
    bool constant_;

public:
    variable_descriptor(const std::string &name, cuda_variable_t key, bool constant);
    bool constant() const;
};
#endif

class GMAC_LOCAL variable_t /*: public variable_descriptor */ {
	CUdeviceptr ptr_;
    size_t size_;
    std::string name_;
public:
	variable_t(CUdeviceptr ptr, size_t size, const std::string &name);
    size_t size() const;
    CUdeviceptr devPtr() const;

    const std::string &get_name() const;
};

class GMAC_LOCAL texture_t /*: public texture_descriptor */ {
protected:
    CUtexref texRef_;
    std::string name_;

public:
	texture_t(CUtexref tex, const std::string &name);

    CUtexref texRef() const;

    const std::string &get_name() const;
};

#if 0
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
#endif

#if 0
typedef std::vector<module_descriptor *> vector_module_descriptor;

class GMAC_LOCAL repository :
    public hal::detail::code::repository {
};
#endif

typedef hal::detail::code::kernel hal_kernel;

class GMAC_LOCAL repository_view :
    public hal::detail::code::repository_view {
protected:
#if 0
    typedef std::map<cuda_variable_t, variable_t> map_variable;
	typedef std::map<cuda_texture_t, texture_t>   map_texture;
    typedef std::map<gmac_kernel_id_t, kernel *>  map_kernel;
#endif

	typedef std::map<std::string, variable_t *> map_variable_name;
	typedef std::map<std::string, texture_t *>  map_texture_name;
    typedef std::map<std::string, kernel *>   map_kernel_name;

#if 0
    map_kernel kernels_;
    map_variable variables_;
	map_variable constants_;
	map_texture textures_;
#endif

    class GMAC_LOCAL module :
        public util::taggeable<> {

        CUmodule mod_;

        map_kernel_name kernelsByName_;
        map_variable_name variablesByName_;
        map_variable_name constantsByName_;
        map_texture_name texturesByName_;

    public:
        module(CUmodule mod) :
            mod_(mod)
        {
        }

        module(CUmodule mod,
               const util::taggeable<> &tags) :
            util::taggeable<>(tags),
            mod_(mod)
        {
        }

        CUmodule operator()()
        {
            return mod_;
        }

        hal_kernel *get_kernel(const std::string &name)
        {
            map_kernel_name::const_iterator it = kernelsByName_.find(name); 
            if (it == kernelsByName_.end()) {
                CUfunction func;
                CUresult res = cuModuleGetFunction(&func, mod_, name.c_str());
                if (res == CUDA_SUCCESS) {
                    kernel *ret = new kernel(func, name);
                    kernelsByName_.insert(map_kernel_name::value_type(name, ret));
                    return ret;
                } else {
                    return NULL;
                }
            }
            return it->second;
        }

        variable_t *get_variable(const std::string &name)
        {
            map_variable_name::const_iterator it = variablesByName_.find(name); 
            if (it == variablesByName_.end()) {
                CUdeviceptr ptr;
                size_t size;
                CUresult res = cuModuleGetGlobal(&ptr, &size, mod_, name.c_str());
                if (res == CUDA_SUCCESS) {
                    variable_t *ret = new variable_t(ptr, size, name);
                    variablesByName_.insert(map_variable_name::value_type(name, ret));
                    return ret;
                } else {
                    return NULL;
                }
            }
            return it->second;
        }

        variable_t *get_constant(const std::string &name)
        {
            map_variable_name::const_iterator it = constantsByName_.find(name); 
            if (it == constantsByName_.end()) {
                CUdeviceptr ptr;
                size_t size;
                CUresult res = cuModuleGetGlobal(&ptr, &size, mod_, name.c_str());
                if (res == CUDA_SUCCESS) {
                    variable_t *ret = new variable_t(ptr, size, name);
                    variablesByName_.insert(map_variable_name::value_type(name, ret));
                    return ret;
                } else {
                    return NULL;
                }
            }
            return it->second;
        }

        texture_t *get_texture(const std::string &name)
        {
            map_texture_name::const_iterator it = texturesByName_.find(name); 
            if (it == texturesByName_.end()) {
                CUtexref tex;
                CUresult res = cuModuleGetTexRef(&tex, mod_, name.c_str());
                if (res == CUDA_SUCCESS) {
                    texture_t *ret = new texture_t(tex, name);
                    texturesByName_.insert(map_texture_name::value_type(name, ret));
                    return ret;
                } else {
                    return NULL;
                }
            }
            return it->second;
        }
    };

	std::vector<module> modules_;
	const void *fatBin_;

public:
#if 0
	repository_view(virt::aspace &as, gmacError_t &err);
#endif
	repository_view(virt::aspace &as, const hal_repository &repo, gmacError_t &err);

	~repository_view();

#if 0
    template <typename T>
    void register_kernels(T &t) const;

    hal_kernel *get_kernel(gmac_kernel_id_t key);
#endif
    const hal_kernel *get_kernel(const std::string &name,
                                 const util::taggeable<>::set_tag &filter = util::taggeable<>::empty);

#if 0
    const variable_t *get_variable(cuda_variable_t key) const;
	const variable_t *get_constant(cuda_variable_t key) const;
#endif
    const variable_t *get_variable(const std::string &name,
                                   const util::taggeable<>::set_tag &filter = util::taggeable<>::empty);
	const variable_t *get_constant(const std::string &name,
                                   const util::taggeable<>::set_tag &filter = util::taggeable<>::empty);
	const texture_t *get_texture(const std::string &name,
                                 const util::taggeable<>::set_tag &filter = util::taggeable<>::empty);
#if 0
    const texture_t  *get_texture(cuda_texture_t key) const;
#endif
};

}}}}

#include "module-impl.h"

#endif
