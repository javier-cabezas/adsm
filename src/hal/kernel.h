#ifndef GMAC_HAL_TYPES_KERNEL_H_
#define GMAC_HAL_TYPES_KERNEL_H_

namespace __impl { namespace hal {

namespace detail {

template <typename I>
class GMAC_LOCAL kernel_t {
public:
    class GMAC_LOCAL config {
    private:
        unsigned ndims_;

    protected:
        config() : ndims_(0) {}
        config(unsigned ndims);

    public:
        unsigned get_ndims() const;

        bool is_valid() const { return ndims_ != 0; }
    };

    class GMAC_LOCAL arg_list {
    public:
        virtual unsigned get_nargs() const = 0;
    };

    class GMAC_LOCAL launch {
    private:
        kernel_t &kernel_;
        typename I::kernel::config &config_;
        typename I::kernel::arg_list &args_;
        typename I::stream &stream_;

    protected:
        typename I::event_ptr event_;

        launch(typename I::kernel &kernel, typename I::kernel::config &config, typename I::kernel::arg_list &args, typename I::stream &stream);

    public:
        typename I::stream &get_stream();
        const typename I::stream &get_stream() const;
 
        const typename I::kernel &get_kernel() const;
        const typename I::kernel::config &get_config() const;
        const typename I::kernel::arg_list &get_arg_list() const;

        virtual typename I::event_ptr execute(list_event<I> &dependencies, gmacError_t &err) = 0;
        virtual typename I::event_ptr execute(typename I::event_ptr event, gmacError_t &err) = 0;
        virtual typename I::event_ptr execute(gmacError_t &err) = 0;

        typename I::event_ptr get_event();
    };

private:
    std::string name_;

protected:
    kernel_t(const std::string &name_);

public:
    const std::string &get_name() const;
    //virtual launch &launch_config(config &config, arg_list &args, typename I::stream &stream) = 0;
};

}

}}

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
