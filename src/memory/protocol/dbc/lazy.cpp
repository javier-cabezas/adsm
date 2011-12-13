#ifdef USE_DBC

#include "memory/protocol/lazy.h"
#include "util/smart_ptr.h"

namespace __dbc { namespace memory { namespace protocol {

lazy_base::lazy_base(bool eager) :
    __impl::memory::protocol::lazy_base(eager)
{
}

lazy_base::~lazy_base()
{
}

__impl::hal::event_ptr
lazy_base::signal_read(block_ptr_impl _block, hostptr_t addr, gmacError_t &err)
{
	REQUIRES(bool(_block));

    lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    __impl::hal::event_ptr ret = parent::signal_read(block, addr, err);

    return ret;
}

__impl::hal::event_ptr
lazy_base::signal_write(block_ptr_impl _block, hostptr_t addr, gmacError_t &err)
{
	REQUIRES(bool(_block));

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::signal_write(block, addr, err);

    ENSURES(block->getState() == __impl::memory::protocol::lazy_types::Dirty);

    return ret;
}

__impl::hal::event_ptr
lazy_base::acquire(block_ptr_impl _block, GmacProtection &prot, gmacError_t &err)
{
	REQUIRES(bool(_block));

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);

    REQUIRES(block->getState() == __impl::memory::protocol::lazy_types::ReadOnly ||
             block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    __impl::hal::event_ptr ret = parent::acquire(block, prot, err);

    ENSURES((prot != GMAC_PROT_READWRITE && prot != GMAC_PROT_WRITE) ||
            block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_ptr
lazy_base::release(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::release(block, err);

    ENSURES(block->getState() == __impl::memory::protocol::lazy_types::ReadOnly ||
            block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_ptr
lazy_base::releaseAll(gmacError_t &err)
{
    __impl::hal::event_ptr ret = parent::releaseAll(err);

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_ptr
lazy_base::toHost(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::toHost(block, err);

    ENSURES(block->getState() != __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_ptr
lazy_base::memset(const block_ptr_impl block, size_t blockOffset, int v, size_t size, gmacError_t &err)
{
	REQUIRES(bool(block));
    REQUIRES(blockOffset + size <= block->size());

    __impl::hal::event_ptr ret = parent::memset(block, blockOffset, v, size, err);

    return ret;
}

__impl::hal::event_ptr
lazy_base::flushDirty(gmacError_t &err)
{
    __impl::hal::event_ptr ret = parent::flushDirty(err);

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_ptr
lazy_base::copyBlockToBlock(block_ptr_impl d, size_t dstOffset, block_ptr_impl s, size_t srcOffset, size_t count, gmacError_t &err)
{
	REQUIRES(bool(d));
	REQUIRES(bool(s));

    lazy_block_ptr_impl dst = __impl::util::static_pointer_cast<lazy_block_impl>(d);
    lazy_block_ptr_impl src = __impl::util::static_pointer_cast<lazy_block_impl>(s);

    REQUIRES(dstOffset + count <= dst->size());
    REQUIRES(srcOffset + count <= src->size());

    state_impl dstState = dst->getState();
    state_impl srcState = src->getState();

    __impl::hal::event_ptr ret = parent::copyBlockToBlock(d, dstOffset, s, srcOffset, count, err);

    ENSURES(dst->getState() == dstState);
    ENSURES(src->getState() == srcState);

    return ret;
}

}}}

#endif // USE_DBC
