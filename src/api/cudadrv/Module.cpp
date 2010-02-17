#include "Module.h"

namespace gmac { namespace gpu {

ModuleDescriptor::ModuleDescriptorVector ModuleDescriptor::Modules;

const char *ModuleDescriptor::pageTableSymbol = "__pageTable";

VariableDescriptor::VariableDescriptor(const char *name, gmacVariable_t key, bool constant) :
    Descriptor<gmacVariable_t>(name, key),
    _constant(constant)
{
}

Variable::Variable(const VariableDescriptor & v, CUmodule mod) :
    VariableDescriptor(v.name(), v.key(), v.constant())
{
    unsigned int tmp;
    CUresult ret = cuModuleGetGlobal(&_ptr, &tmp, mod, name());
    assert(ret == CUDA_SUCCESS);
    _size = tmp;
}

Texture::Texture(const TextureDescriptor & t, CUmodule mod) :
    TextureDescriptor(t.name(), t.key())
{
    assert(cuModuleGetTexRef(&_texRef, mod, name()) == CUDA_SUCCESS);
}

ModuleDescriptor::ModuleDescriptor(const void *fatBin) :
    _fatBin(fatBin),
    _pageTable(NULL)
{
    TRACE("Creating module descriptor: %p", _fatBin);
    Modules.push_back(this);
}

ModuleVector
ModuleDescriptor::createModules()
{
    TRACE("Creating modules");
    ModuleVector modules;

    ModuleDescriptorVector::const_iterator it;
    for (it = Modules.begin(); it != Modules.end(); it++) {
        TRACE("Creating module: %p", (*it)->_fatBin);
        modules.push_back(new Module(*(*it)));
    }
    return modules;
}

Module::Module(const ModuleDescriptor & d) :
    _fatBin(d._fatBin)
{
    TRACE("Module image: %p", _fatBin);
    CUresult res;
    res = cuModuleLoadFatBinary(&_mod, _fatBin);
    assert(res == CUDA_SUCCESS);

    Context * ctx = Context::current();
    ModuleDescriptor::KernelVector::const_iterator k;
    for (k = d._kernels.begin(); k != d._kernels.end(); k++) {
        Kernel * kernel = new Kernel(*k, _mod);
        ctx->kernel(k->key(), kernel);
    }

    ModuleDescriptor::VariableVector::const_iterator v;
    for (v = d._variables.begin(); v != d._variables.end(); v++) {
        _variables.insert(VariableMap::value_type(v->key(), Variable(*v, _mod)));
    }

    for (v = d._constants.begin(); v != d._constants.end(); v++) {
        _constants.insert(VariableMap::value_type(v->key(), Variable(*v, _mod)));
    }

    ModuleDescriptor::TextureVector::const_iterator t;
    for (t = d._textures.begin(); t != d._textures.end(); t++) {
        _textures.insert(TextureMap::value_type(t->key(), Texture(*t, _mod)));
    }

    if (d._pageTable != NULL) {
        VariableMap::iterator it;
        it = _variables.find(d._pageTable->key());
        if (it == _variables.end()) {
            it = _constants.find(d._pageTable->key());
            assert(it != _constants.end());
            _pageTable = &it->second;
        }
    }
}

Module::~Module() {
    assert(cuModuleUnload(_mod) == CUDA_SUCCESS);
}

}}
