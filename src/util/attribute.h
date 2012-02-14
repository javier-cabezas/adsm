#ifndef GMAC_UTIL_ATTRIBUTE_H_
#define GMAC_UTIL_ATTRIBUTE_H_

#include "atomics.h"

namespace __impl { namespace util {

template <typename T>
class GMAC_LOCAL attributes {
protected:
    static Atomic id_;

    typedef std::vector<void *> vector_attributes;
    vector_attributes attributes_;

public:
    typedef Atomic attribute_id;

    static attribute_id register_attribute()
    {
        return AtomicInc(id_) - 1;
    }

    attributes() :
        attributes_(id_, NULL)
    {
    }

    template <typename S = void>
    void set_attribute(attribute_id id, S *attribute)
    {
        attributes_[id] = attribute;
    }

    template <typename S = void>
    S *get_attribute(attribute_id id)
    {
        return reinterpret_cast<S *>(attributes_[id]);
    }
};

template <typename T>
Atomic attributes<T>::id_ = 0;

}}

#endif // GMAC_UTIL_ATTRIBUTE_H_

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
