#include "hpe/init.h"

#include "module.h"
#include "hal/cuda/types.h"

namespace __impl { namespace hal {
        
#if 0
extern cuda::code::map_context_repository Modules_;
#endif

namespace cuda { namespace code {

variable_t::variable_t(CUdeviceptr ptr, size_t size, const std::string &name) :
    ptr_(ptr),
    size_(size),
    name_(name)
{
#if 0
#if CUDA_VERSION > 3010
    size_t tmp;
#else
    unsigned int tmp;
#endif
    TRACE(LOCAL, "Creating new accelerator variable: %s", v.get_name().c_str());
    CUresult ret = cuModuleGetGlobal(&ptr_, &tmp, mod, get_name().c_str());
    ASSERTION(ret == CUDA_SUCCESS);
    size_ = tmp;
#endif
}

texture_t::texture_t(CUtexref tex, const std::string &name) :
    texRef_(tex),
    name_(name)
{
}

#if 0
module_descriptor::module_descriptor(const void *fatBin) :
    fatBin_(fatBin)
{
    TRACE(LOCAL, "Creating module descriptor: %p", fatBin_);
    ModuleDescriptors_.push_back(this);
}

repository *
module_descriptor::create_modules()
{
    TRACE(GLOBAL, "Creating modules");

    repository *ptr = new repository(ModuleDescriptors_);
#if 0
    vector_module_descriptor::const_iterator it;
    for (it = ModuleDescriptors_.begin(); it != ModuleDescriptors_.end(); it++) {
        TRACE(GLOBAL, "Creating module: %p", (*it)->fatBin_);
        modules.push_back(new cuda::module(*(*it)));
    }
#endif
    return ptr;
}
#endif

static const int CUDA_MAGIC = 0x466243b1;

struct GMAC_LOCAL FatBinDesc {
    int magic; int v; const unsigned long long* data; char* f;
};

repository_view::repository_view(virt::aspace &as, const hal_repository &repo, hal::error &err)
{
    as.set();

    for (auto &file : repo.get_files()) {
        CUmodule cumod;

        // TODO: add support for flags
        CUresult res = cuModuleLoad(&cumod, file.get_path().c_str());

        err = error_to_hal(res);
        if (err != hal::error::HAL_SUCCESS) return;

        module mod(cumod, file);
        modules_.push_back(mod);
    }

    for (auto &buffer : repo.get_buffers()) {
        CUmodule cumod;

        // TODO: add support for flags
        CUresult res = cuModuleLoadData(&cumod, buffer.get_ptr());

        err = error_to_hal(res);
        if (err != hal::error::HAL_SUCCESS) return;

        module mod(cumod, buffer);
        modules_.push_back(cumod);
    }

    for (auto &handle : repo.get_handles()) {
        CUmodule cumod;

        const void *h = handle.get_handle();
        FatBinDesc *desc = (FatBinDesc *)h;
        // TODO: check when this is necessary
        if (desc->magic == CUDA_MAGIC) {
            h = desc->data;
        }

        // TODO: add support for flags
        CUresult res = cuModuleLoadData(&cumod, h);

        err = error_to_hal(res);
        if (err != hal::error::HAL_SUCCESS) return;

        module mod(cumod, handle);
        modules_.push_back(mod);
    }

#if 0
    for (auto mod : modules_) {
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

        modules_.push_back(mod);
    }
#endif
}

repository_view::~repository_view()
{
    modules_.clear();

#if 0
    map_kernel::iterator it;
    for (it = kernels_.begin(); it != kernels_.end(); ++it) {
        delete it->second;
    }
#ifdef CALL_CUDA_ON_DESTRUCTION
    std::vector<CUmodule>::const_iterator m;
    for(m = modules_.begin(); m != modules_.end(); ++m) {
        CUresult ret = cuModuleUnload(*m);
        ASSERTION(ret == CUDA_SUCCESS);
    }
    modules_.clear();
#endif
#endif

    // TODO: remove objects from maps
#if 0
    variables_.clear();
    constants_.clear();
    textures_.clear();
#endif
}

#if 0
hal_kernel *
repository_view::get_kernel(gmac_kernel_id_t key)
{
    map_kernel::const_iterator k;
    k = kernels_.find(key);
    if (k == kernels_.end()) return NULL;
    return k->second;
}

const variable_t *
repository_view::get_constant(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = constants_.find(key);
    if(v == constants_.end()) return NULL;
    return &v->second;
}

const variable_t *
repository_view::get_variable(cuda_variable_t key) const
{
    map_variable::const_iterator v;
    v = variables_.find(key);
    if(v == variables_.end()) return NULL;
    return &v->second;
}

const texture_t *
repository_view::get_texture(cuda_texture_t key) const
{
    map_texture::const_iterator t;
    t = textures_.find(key);
    if(t == textures_.end()) return NULL;
    return &t->second;
}

#endif

const hal_kernel *
repository_view::get_kernel(const std::string &name,
                            const util::taggeable<>::set_tag &tags)
{
    const hal_kernel *ret = NULL;
    for (auto &mod : modules_) {
        if (mod.has_tags(tags)) {
            ret = mod.get_kernel(name);
            if (ret != NULL) {
                return ret;
            }
        }
    }
    return NULL;
}

const variable_t *
repository_view::get_constant(const std::string &name,
                              const util::taggeable<>::set_tag &tags)
{
    const variable_t *ret = NULL;
    for (module &mod : modules_) {
        if (mod.has_tags(tags)) {
            ret = mod.get_constant(name);
            if (ret != NULL) {
                return ret;
            }
        }
    }
    return NULL;
}

const variable_t *
repository_view::get_variable(const std::string &name,
                              const util::taggeable<>::set_tag &tags)
{
    const variable_t *ret = NULL;
    for (module &mod : modules_) {
        if (mod.has_tags(tags)) {
            ret = mod.get_variable(name);
            if (ret != NULL) {
                return ret;
            }
        }
    }
    return NULL;
}

const texture_t *
repository_view::get_texture(const std::string &name,
                              const util::taggeable<>::set_tag &tags)
{
    const texture_t *ret = NULL;
    for (module &mod : modules_) {
        if (mod.has_tags(tags)) {
            ret = mod.get_texture(name);
            if (ret != NULL) {
                return ret;
            }
        }
    }
    return NULL;
}

}}}}
