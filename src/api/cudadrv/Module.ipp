#ifndef __API_CUDADRV_MODULE_IPP_
#define __API_CUDADRV_MODULE_IPP_

namespace gmac { namespace gpu {

inline bool
VariableDescriptor::constant() const
{
    return _constant;
}

inline size_t
Variable::size() const
{
    return _size;
}

inline CUdeviceptr
Variable::devPtr() const
{
    return _ptr;
}

inline CUtexref
Texture::texRef() const
{
    return _texRef;
}

inline
const VariableDescriptor &
ModuleDescriptor::pageTable() const
{
    return *_pageTable;
}

inline
void
ModuleDescriptor::add(gmac::KernelDescriptor & k)
{
    _kernels.push_back(k);
}

inline
void
ModuleDescriptor::add(VariableDescriptor & v)
{
    if (v.constant()) {
        _constants.push_back(v);
    } else {
        _variables.push_back(v);
    }
}

inline
void
ModuleDescriptor::add(TextureDescriptor & t)
{
    _textures.push_back(t);
}

inline const Variable *
Module::constant(gmacVariable_t key) const
{
    VariableMap::const_iterator v;
    v = _constants.find(key);
    if(v == _constants.end()) return NULL;
    return &v->second;
}

inline const Variable *
Module::variable(gmacVariable_t key) const
{
    VariableMap::const_iterator v;
    v = _variables.find(key);
    if(v == _variables.end()) return NULL;
    return &v->second;
}

inline const Texture *
Module::texture(gmacTexture_t key) const
{
    TextureMap::const_iterator t;
    t = _textures.find(key);
    if(t == _textures.end()) return NULL;
    return &t->second;
}

}}

#endif
