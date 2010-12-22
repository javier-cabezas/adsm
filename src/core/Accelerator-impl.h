#ifndef GMAC_CORE_ACCELERATOR_IMPL_H_
#define GMAC_CORE_ACCELERATOR_IMPL_H_

namespace __impl { namespace core {

inline unsigned
Accelerator::id() const
{
    return id_;
}

inline unsigned
Accelerator::busId() const
{
	return busId_;
}

inline unsigned
Accelerator::busAccId() const
{
	return busAccId_;
}

inline bool
Accelerator::integrated() const
{
	return integrated_;
}

}}

#endif
