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

#ifndef GMAC_HAL_OPENCL_MODULE_H_
#define GMAC_HAL_OPENCL_MODULE_H_

#include <CL/cl.h>

#include <list>
#include <map>
#include <string>
#include <vector>

#include "config/common.h"
#include "config/config.h"

#include "util/descriptor.h"
#include "util/UniquePtr.h"
#include "util/stl/locked_map.h"

#include "types.h"

namespace __impl { namespace hal { namespace opencl {

typedef util::descriptor<gmac_kernel_id_t> kernel_descriptor;

class code_repository;
typedef util::stl::locked_map<context_t *, code_repository *> map_context_repository;

class GMAC_LOCAL module_descriptor {
	friend class code_repository;

protected:
    typedef std::vector<module_descriptor *> vector_module_descriptor;
    static vector_module_descriptor ModuleDescriptors_;
	const void *fatBin_;

    typedef std::vector<kernel_descriptor>   vector_kernel;

    vector_kernel kernels_;

public:
    module_descriptor(const void * fatBin);

    void add(kernel_descriptor   &k);

    static code_repository *create_modules();
};

typedef std::vector<module_descriptor *> vector_module_descriptor;

class GMAC_LOCAL code_repository :
    public hal::detail::code_repository<device, backend_traits, implementation_traits> {
protected:
    typedef std::map<gmac_kernel_id_t, kernel_t *> map_kernel;
    typedef std::map<std::string, kernel_t *> map_kernel_name;

    map_kernel kernels_;
    map_kernel_name kernelsByName_;

public:
	code_repository(const vector_module_descriptor & dVector);
	~code_repository();

    template <typename T>
    void register_kernels(T &t) const;

    const kernel_t *kernel(gmac_kernel_id_t key) const;
    const kernel_t *kernelByName(const std::string &name) const;
};

}}}

//#include "module-impl.h"

#endif
