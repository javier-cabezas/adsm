#ifdef USE_DBC

#include "memory/protocol/Lazy.h"
#include "util/UniquePtr.h"

namespace __dbc { namespace memory { namespace protocol {

lazy_base::lazy_base(bool eager) :
    __impl::memory::protocol::lazy_base(eager)
{
}

lazy_base::~lazy_base()
{
}

gmacError_t
lazy_base::signal_read(block_ptr_impl _block, hostptr_t addr)
{
	REQUIRES(_block);

    lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    gmacError_t ret = parent::signal_read(block, addr);

    return ret;
}

gmacError_t
lazy_base::signal_write(block_ptr_impl _block, hostptr_t addr)
{
	REQUIRES(_block);

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    gmacError_t ret = parent::signal_write(block, addr);

    ENSURES(block->getState() == __impl::memory::protocol::lazy_types::Dirty);

    return ret;
}

gmacError_t
lazy_base::acquire(block_ptr_impl _block, GmacProtection &prot)
{
	REQUIRES(_block);

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);

    REQUIRES(block->getState() == __impl::memory::protocol::lazy_types::ReadOnly ||
             block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    gmacError_t ret = parent::acquire(block, prot);

    ENSURES((prot != GMAC_PROT_READWRITE && prot != GMAC_PROT_WRITE) ||
            block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

gmacError_t
lazy_base::release(block_ptr_impl _block)
{
	REQUIRES(_block);
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    gmacError_t ret = parent::release(block);

    ENSURES(block->getState() == __impl::memory::protocol::lazy_types::ReadOnly ||
            block->getState() == __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

gmacError_t
lazy_base::releaseAll()
{
    gmacError_t ret = parent::releaseAll();

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

gmacError_t
lazy_base::toHost(block_ptr_impl _block)
{
	REQUIRES(_block);
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    gmacError_t ret = parent::toHost(block);

    ENSURES(block->getState() != __impl::memory::protocol::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_t
lazy_base::memset(const block_ptr_impl block, size_t blockOffset, int v, size_t size, gmacError_t &err)
{
	REQUIRES(block);
    REQUIRES(blockOffset + size <= block->size());

    __impl::hal::event_t ret = parent::memset(block, blockOffset, v, size, err);

    return ret;
}

gmacError_t
lazy_base::flushDirty()
{
    gmacError_t ret = parent::flushDirty();

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_t
lazy_base::copyBlockToBlock(block_ptr_impl d, size_t dstOffset, block_ptr_impl s, size_t srcOffset, size_t count, gmacError_t &err)
{
	REQUIRES(d);
	REQUIRES(s);

    lazy_block_ptr_impl dst = __impl::util::static_pointer_cast<lazy_block_impl>(d);
    lazy_block_ptr_impl src = __impl::util::static_pointer_cast<lazy_block_impl>(s);

    REQUIRES(dstOffset + count <= dst->size());
    REQUIRES(srcOffset + count <= src->size());

    state_impl dstState = dst->getState();
    state_impl srcState = src->getState();

    __impl::hal::event_t ret = parent::copyBlockToBlock(d, dstOffset, s, srcOffset, count, err);

    ENSURES(dst->getState() == dstState);
    ENSURES(src->getState() == srcState);

    return ret;
}

}}}

#endif // USE_DBC
