#include "hpe/init.h"

#include "module.h"

namespace __impl { namespace hal {
        
extern cuda::map_context_repository Modules_;

namespace cuda {

module_descriptor::vector_module_descriptor module_descriptor::ModuleDescriptors_;

variable_descriptor::variable_descriptor(const std::string &name, cuda_variable_t key, bool constant) :
    util::descriptor<cuda_variable_t>(name, key),
    constant_(constant)
{
}

variable_t::variable_t(const variable_descriptor & v, CUmodule mod) :
    variable_descriptor(v.get_name(), v.get_key(), v.constant())
{
#if CUDA_VERSION > 3010
    size_t tmp;
#else
    unsigned int tmp;
#endif
    TRACE(LOCAL, "Creating new accelerator variable: %s", v.get_name().c_str());
    CUresult ret = cuModuleGetGlobal(&ptr_, &tmp, mod, get_name().c_str());
    ASSERTION(ret == CUDA_SUCCESS);
    size_ = tmp;
}

texture_t::texture_t(const texture_descriptor & t, CUmodule mod) :
    texture_descriptor(t.get_name(), t.get_key())
{
    CUresult ret = cuModuleGetTexRef(&texRef_, mod, get_name().c_str());
    ASSERTION(ret == CUDA_SUCCESS);
}

module_descriptor::module_descriptor(const void *fatBin) :
    fatBin_(fatBin)
{
    TRACE(LOCAL, "Creating module descriptor: %p", fatBin_);
    ModuleDescriptors_.push_back(this);
}

code_repository *
module_descriptor::create_modules()
{
    TRACE(GLOBAL, "Creating modules");

    code_repository *ptr = new code_repository(ModuleDescriptors_);
#if 0
    vector_module_descriptor::const_iterator it;
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); it++) {
        TRACE(GLOBAL, "Creating module: %p", (*it)->fatBin_);
        modules.push_back(new cuda::module(*(*it)));
    }
#endif
    return ptr;
}

static const int CUDA_MAGIC = 0x466243b1;

struct GMAC_LOCAL FatBinDesc {
    int magic; int v; const unsigned long long* data; char* f;
};

code_repository::code_repository(const vector_module_descriptor & dVector)
{
    CUmodule mod;
    vector_module_descriptor::const_iterator it;
    for (it = dVector.begin(); it != dVector.end(); ++it) {
        module_descriptor &d = *(*it);
        const void *fatBin_ = d.fatBin_;
        TRACE(LOCAL, "module image: %p", fatBin_);
        CUresult res;

        FatBinDesc *desc = (FatBinDesc *)fatBin_;
        if (desc->magic == CUDA_MAGIC) {
            res = cuModuleLoadFatBinary(&mod, desc->data);
        } else {
            res = cuModuleLoadFatBinary(&mod, fatBin_);
        }
        CFATAL(res == CUDA_SUCCESS, "Error loading module: %d", res);

        module_descriptor::vector_kernel::const_iterator k;
        for (k = d.kernels_.begin(); k != d.kernels_.end(); ++k) {
            TRACE(LOCAL, "Registering kernel: %s", k->get_name().c_str());
            CUfunction func;
            res = cuModuleGetFunction(&func, mod, k->get_name().c_str());
            kernels_.insert(map_kernel::value_type(k->get_key(), new kernel(func, k->get_name())));
        }

        module_descriptor::vector_variable::const_iterator v;
        for (v = d.variables_.begin(); v != d.variables_.end(); ++v) {
            map_variable_name::const_iterator f;
            f = variablesByName_.find(v->get_name());
            if (f != variablesByName_.end()) {
                FATAL("variable_t already registered: %s", v->get_name().c_str());
            } else {
                TRACE(LOCAL, "Registering variable: %s", v->get_name().c_str());
                variables_.insert(map_variable::value_type(v->get_key(), variable_t(*v, mod)));
                variablesByName_.insert(map_variable_name::value_type(v->get_name(), variable_t(*v, mod)));
            }
        }

        for (v = d.constants_.begin(); v != d.constants_.end(); ++v) {
            map_variable_name::const_iterator f;
            f = constantsByName_.find(v->get_name());
            if (f != constantsByName_.end()) {
                FATAL("Constant already registered: %s", v->get_name().c_str());
            }
            TRACE(LOCAL, "Registering constant: %s", v->get_name().c_str());
            constants_.insert(map_variable::value_type(v->get_key(), variable_t(*v, mod)));
            constantsByName_.insert(map_variable_name::value_type(v->get_name(), variable_t(*v, mod)));
        }

        module_descriptor::vector_texture::const_iterator t;
        for (t = d.textures_.begin(); t != d.textures_.end(); ++t) {
            textures_.insert(map_texture::value_type(t->get_key(), texture_t(*t, mod)));
        }

        mods_.push_back(mod);
    }
}

code_repository::~code_repository()
{
    map_kernel::iterator it;
    for (it = kernels_.begin(); it != kernels_.end(); ++it) {
        delete it->second;
    }
#ifdef CALL_CUDA_ON_DESTRUCTION
    std::vector<CUmodule>::const_iterator m;
    for(m = mods_.begin(); m != mods_.end(); ++m) {
        CUresult ret = cuModuleUnload(*m);
        ASSERTION(ret == CUDA_SUCCESS);
    }
    mods_.clear();
#endif

    // TODO: remove objects from maps
#if 0
    variables_.clear();
    constants_.clear();
    textures_.clear();
#endif
}

hal_kernel *
code_repository::get_kernel(gmac_kernel_id_t key)
{
    map_kernel::const_iterator k;
    k = kernels_.find(key);
    if(k == kernels_.end()) return NULL;
    return k->second;
}

hal_kernel *
code_repository::get_kernel(const std::string &name)
{
    FATAL("Not implemented");
    return NULL;
}

const variable_t *
code_repository::constant(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = constants_.find(key);
    if(v == constants_.end()) return NULL;
    return &v->second;
}

const variable_t *
code_repository::variable(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = variables_.find(key);
    if(v == variables_.end()) return NULL;
    return &v->second;
}

const variable_t *
code_repository::constantByName(const std::string &name) const
{
    map_variable_name::const_iterator v;
    v = constantsByName_.find(name);
    if(v == constantsByName_.end()) return NULL;
    return &v->second;
}

const variable_t *
code_repository::variableByName(const std::string &name) const
{
    map_variable_name::const_iterator v;
    v = variablesByName_.find(name);
    if(v == variablesByName_.end()) return NULL;
    return &v->second;
}

const texture_t *
code_repository::texture(cuda_texture_t key) const
{
    map_texture::const_iterator t;
    t = textures_.find(key);
    if(t == textures_.end()) return NULL;
    return &t->second;
}

}}}
