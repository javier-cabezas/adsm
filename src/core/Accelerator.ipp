
#ifndef GMAC_CORE_ACCELERATOR_IPP_
#define GMAC_CORE_ACCELERATOR_IPP_

namespace gmac { namespace core {

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
Accelerator::busDevId() const
{
	return busDevId_;
}

inline bool
Accelerator::integrated() const
{
	return integrated_;
}

}}

#endif
