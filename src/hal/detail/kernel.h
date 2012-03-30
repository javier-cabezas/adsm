#ifndef GMAC_HAL_TYPES_KERNEL_H_
#define GMAC_HAL_TYPES_KERNEL_H_

namespace __impl { namespace hal {

namespace detail { namespace code {

class GMAC_LOCAL kernel {
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
        kernel &kernel_;

    protected:
        event_ptr event_;

        launch(kernel &kernel);

    public:
        const kernel &get_kernel() const;
        
        virtual event_ptr execute(list_event &dependencies, gmacError_t &err) = 0;
        virtual event_ptr execute(event_ptr event, gmacError_t &err) = 0;
        virtual event_ptr execute(gmacError_t &err) = 0;

        event_ptr get_event();
    };

private:
    std::string name_;

protected:
    kernel(const std::string &name_);

public:
    const std::string &get_name() const;
    //virtual launch &launch_config(config &config, arg_list &args, typename I::stream &stream) = 0;
};

}}

}}

#endif /* KERNEL_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
