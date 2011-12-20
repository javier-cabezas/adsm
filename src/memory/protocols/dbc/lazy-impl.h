#ifndef GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_
#define GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

namespace __dbc { namespace memory { namespace protocols {

template <typename T>
lazy<T>::lazy(bool eager) :
    parent(eager)
{
}

template <typename T>
lazy<T>::~lazy()
{
}


}}}

#endif //GMAC_MEMORY_PROTOCOL_DBC_LAZY_IMPL_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
