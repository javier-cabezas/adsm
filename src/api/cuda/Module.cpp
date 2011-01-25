#include "gmac/init.h"

#include "Module.h"
#include "Mode.h"

namespace __impl { namespace cuda {

ModuleDescriptor::ModuleDescriptorVector ModuleDescriptor::Modules_;

VariableDescriptor::VariableDescriptor(const char *name, gmacVariable_t key, bool constant) :
    core::Descriptor<gmacVariable_t>(name, key),
    constant_(constant)
{
}

Variable::Variable(const VariableDescriptor & v, CUmodule mod) :
    VariableDescriptor(v.name(), v.key(), v.constant())
{
#if CUDA_VERSION > 3010
    size_t tmp;
#else
    unsigned int tmp;
#endif
    TRACE(LOCAL, "Creating new accelerator variable: %s", v.name());
    CUresult ret = cuModuleGetGlobal(&ptr_, &tmp, mod, name());
    ASSERTION(ret == CUDA_SUCCESS);
    size_ = tmp;
}

Texture::Texture(const TextureDescriptor & t, CUmodule mod) :
    TextureDescriptor(t.name(), t.key())
{
    CUresult ret = cuModuleGetTexRef(&texRef_, mod, name());
    ASSERTION(ret == CUDA_SUCCESS);
}

ModuleDescriptor::ModuleDescriptor(const void *fatBin) :
    fatBin_(fatBin)
{
    TRACE(LOCAL, "Creating module descriptor: %p", fatBin_);
    Modules_.push_back(this);
}

ModuleDescriptor::~ModuleDescriptor()
{
    kernels_.clear();
    variables_.clear();
    constants_.clear();
    textures_.clear();
}

ModuleVector
ModuleDescriptor::createModules()
{
    TRACE(GLOBAL, "Creating modules");
    ModuleVector modules;

    ModuleDescriptorVector::const_iterator it;
    for (it = Modules_.begin(); it != Modules_.end(); it++) {
        TRACE(GLOBAL, "Creating module: %p", (*it)->fatBin_);
        modules.push_back(new cuda::Module(*(*it)));
    }
    return modules;
}

Module::Module(const ModuleDescriptor & d) :
    fatBin_(d.fatBin_)
{
    TRACE(LOCAL, "Module image: %p", fatBin_);
    CUresult res;
    res = cuModuleLoadFatBinary(&mod_, fatBin_);
    CFATAL(res == CUDA_SUCCESS, "Error loading module: %d", res);

    ModuleDescriptor::KernelVector::const_iterator k;
    for (k = d.kernels_.begin(); k != d.kernels_.end(); k++) {
        Kernel * kernel = new cuda::Kernel(*k, mod_);
        kernels_.insert(KernelMap::value_type(k->key(), kernel));
    }

    ModuleDescriptor::VariableVector::const_iterator v;
    for (v = d.variables_.begin(); v != d.variables_.end(); v++) {
        variables_.insert(VariableMap::value_type(v->key(), Variable(*v, mod_)));
        variablesByName_.insert(VariableNameMap::value_type(std::string(v->name()), Variable(*v, mod_)));
    }

    for (v = d.constants_.begin(); v != d.constants_.end(); v++) {
        constants_.insert(VariableMap::value_type(v->key(), Variable(*v, mod_)));
        constantsByName_.insert(VariableNameMap::value_type(std::string(v->name()), Variable(*v, mod_)));
    }

    ModuleDescriptor::TextureVector::const_iterator t;
    for (t = d.textures_.begin(); t != d.textures_.end(); t++) {
        textures_.insert(TextureMap::value_type(t->key(), Texture(*t, mod_)));
    }

}

Module::~Module()
{
    CUresult ret = cuModuleUnload(mod_);
    ASSERTION(ret == CUDA_SUCCESS);

    // TODO: remove objects from maps
#if 0
    variables_.clear();
    constants_.clear();
    textures_.clear();
#endif

    KernelMap::iterator i;
    for(i = kernels_.begin(); i != kernels_.end(); i++) delete i->second;
    kernels_.clear();
}

void Module::registerKernels(Mode &mode) const
{
    KernelMap::const_iterator k;
    for (k = kernels_.begin(); k != kernels_.end(); k++) {
        mode.kernel(k->first, *k->second);
    }
}

}}
