#ifndef GMAC_HAL_TYPES_KERNEL_H_
#define GMAC_HAL_TYPES_KERNEL_H_

namespace __impl { namespace hal {

namespace detail {

template <typename B, typename I>
class GMAC_LOCAL kernel_t {
public:
    class GMAC_LOCAL config {
    private:
        unsigned ndims_;

    protected:
        config(unsigned ndims);

    public:
        unsigned get_ndims() const;
        virtual unsigned get_nargs() const = 0;

        // virtual const typename B::kernel_config &get_dims_global() const = 0;
        // virtual const typename B::kernel_config &get_dims_group() const = 0;

        // virtual gmacError_t set_arg(const void *arg, size_t size, unsigned index) = 0;
    };

    class GMAC_LOCAL launch {
    private:
        kernel_t &kernel_;
        typename I::kernel::config config_;
        typename I::stream &stream_;

    protected:
        launch(typename I::kernel &kernel, typename I::kernel::config &conf, typename I::stream &stream);

    public:
        typename I::stream &get_stream();
        const typename I::stream &get_stream() const;
 
        const typename I::kernel &get_kernel() const;
        const typename I::kernel::config &get_config() const;

        virtual typename I::event execute(list_event<I> &dependencies, gmacError_t &err) = 0;
        virtual typename I::event execute(typename I::event event, gmacError_t &err) = 0;
        virtual typename I::event execute(gmacError_t &err) = 0;
    };

private:
    typename B::kernel kernel_;
    std::string name_;

protected:
    kernel_t(typename B::kernel kernel, const std::string &name_);

public:
    typename B::kernel &operator()();
    const typename B::kernel &operator()() const;

    const std::string &get_name() const;
    virtual launch &launch_config(config &config, typename I::stream &stream) = 0;
};

}

}}

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
