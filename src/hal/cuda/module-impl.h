#ifndef GMAC_API_CUDA_HPE_MODULE_IMPL_H_
#define GMAC_API_CUDA_HPE_MODULE_IMPL_H_

namespace __impl { namespace hal { namespace cuda {

inline bool
variable_descriptor::constant() const
{
    return constant_;
}

inline size_t
variable_t::size() const
{
    return size_;
}

inline CUdeviceptr
variable_t::devPtr() const
{
    return ptr_;
}

inline CUtexref
texture_t::texRef() const
{
    return texRef_;
}

inline
void
module_descriptor::add(kernel_descriptor & k)
{
    kernels_.push_back(k);
}

inline
void
module_descriptor::add(variable_descriptor & v)
{
    if (v.constant()) {
        constants_.push_back(v);
    } else {
        variables_.push_back(v);
    }
}

inline
void
module_descriptor::add(texture_descriptor &t)
{
    textures_.push_back(t);
}

inline kernel_t *
code_repository::get_kernel(gmac_kernel_id_t key)
{
    map_kernel::const_iterator k;
    k = kernels_.find(key);
    if(k == kernels_.end()) return NULL;
    return k->second;
}

inline kernel_t *
code_repository::get_kernel(const std::string &name)
{
    FATAL("Not implemented");
    return NULL;
}

inline const variable_t *
code_repository::constant(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = constants_.find(key);
    if(v == constants_.end()) return NULL;
    return &v->second;
}

inline const variable_t *
code_repository::variable(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = variables_.find(key);
    if(v == variables_.end()) return NULL;
    return &v->second;
}

inline const variable_t *
code_repository::constantByName(const std::string &name) const
{
    map_variable_name::const_iterator v;
    v = constantsByName_.find(name);
    if(v == constantsByName_.end()) return NULL;
    return &v->second;
}

inline const variable_t *
code_repository::variableByName(const std::string &name) const
{
    map_variable_name::const_iterator v;
    v = variablesByName_.find(name);
    if(v == variablesByName_.end()) return NULL;
    return &v->second;
}

inline const texture_t *
code_repository::texture(cuda_texture_t key) const
{
    map_texture::const_iterator t;
    t = textures_.find(key);
    if(t == textures_.end()) return NULL;
    return &t->second;
}

#if 0
template <typename T>
void
code_repository::register_kernels(T &t) const
{
    map_kernel::const_iterator k;
    for (k = kernels_.begin(); k != kernels_.end(); k++) {
        t.register_kernel(k->first, *k->second);
    }
}
#endif

}}}

#endif
