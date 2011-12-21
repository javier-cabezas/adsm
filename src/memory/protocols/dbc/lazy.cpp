#ifdef USE_DBC

#include "memory/protocols/lazy.h"
#include "util/smart_ptr.h"

namespace __dbc { namespace memory { namespace protocols {

lazy_base::lazy_base(bool eager) :
    __impl::memory::protocols::lazy_base(eager)
{
}

lazy_base::~lazy_base()
{
}

__impl::hal::event_ptr
lazy_base::signal_read(block_ptr_impl _block, host_ptr addr, gmacError_t &err)
{
	REQUIRES(bool(_block));

    lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    //REQUIRES(block.getState() == __impl::memory::protocol::lazy::Invalid);

    __impl::hal::event_ptr ret = parent::signal_read(block, addr, err);

    return ret;
}

__impl::hal::event_ptr
lazy_base::signal_write(block_ptr_impl _block, host_ptr addr, gmacError_t &err)
{
	REQUIRES(bool(_block));

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::signal_write(block, addr, err);

    ENSURES(block->get_state() == __impl::memory::protocols::lazy_types::Dirty);

    return ret;
}

__impl::hal::event_ptr
lazy_base::acquire(block_ptr_impl _block, GmacProtection prot, gmacError_t &err)
{
	REQUIRES(bool(_block));

	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);

    REQUIRES(block->get_state() == __impl::memory::protocols::lazy_types::ReadOnly ||
             block->get_state() == __impl::memory::protocols::lazy_types::Invalid);

    __impl::hal::event_ptr ret = parent::acquire(block, prot, err);

    ENSURES((prot != GMAC_PROT_READWRITE && prot != GMAC_PROT_WRITE) ||
            block->get_state() == __impl::memory::protocols::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_ptr
lazy_base::release(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::release(block, err);

    ENSURES(block->get_state() == __impl::memory::protocols::lazy_types::ReadOnly ||
            block->get_state() == __impl::memory::protocols::lazy_types::Invalid);

    return ret;
}

__impl::hal::event_ptr
lazy_base::release_all(gmacError_t &err)
{
    __impl::hal::event_ptr ret = parent::release_all(err);

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_ptr
lazy_base::map_to_device(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
	__impl::hal::event_ptr ret = parent::map_to_device(block, err);

	ENSURES(block->get_state() == __impl::memory::protocols::lazy_types::Dirty);

	return ret;
}

__impl::hal::event_ptr
lazy_base::unmap_from_device(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
	__impl::hal::event_ptr ret = parent::unmap_from_device(block, err);

	ENSURES(block->get_state() == __impl::memory::protocols::lazy_types::Dirty ||
			block->get_state() == __impl::memory::protocols::lazy_types::ReadOnly);

	return ret;
}

__impl::hal::event_ptr
lazy_base::remove_block(block_ptr_impl block, gmacError_t &err)
{
	REQUIRES(bool(block));
	__impl::hal::event_ptr ret = parent::remove_block(block, err);

	return ret;
}

__impl::hal::event_ptr
lazy_base::to_host(block_ptr_impl _block, gmacError_t &err)
{
	REQUIRES(bool(_block));
	lazy_block_ptr_impl block = __impl::util::static_pointer_cast<lazy_block_impl>(_block);
    __impl::hal::event_ptr ret = parent::to_host(block, err);

    ENSURES(block->get_state() != __impl::memory::protocols::lazy_types::Invalid);

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
lazy_base::flush_dirty(gmacError_t &err)
{
    __impl::hal::event_ptr ret = parent::flush_dirty(err);

    ENSURES(parent::dbl_.size() == 0);

    return ret;
}

__impl::hal::event_ptr
lazy_base::copy_block_to_block(block_ptr_impl d, size_t dstOffset, block_ptr_impl s, size_t srcOffset, size_t count, gmacError_t &err)
{
	REQUIRES(bool(d));
	REQUIRES(bool(s));

    lazy_block_ptr_impl dst = __impl::util::static_pointer_cast<lazy_block_impl>(d);
    lazy_block_ptr_impl src = __impl::util::static_pointer_cast<lazy_block_impl>(s);

    REQUIRES(dstOffset + count <= dst->size());
    REQUIRES(srcOffset + count <= src->size());

    state_impl dstState = dst->get_state();
    state_impl srcState = src->get_state();

    __impl::hal::event_ptr ret = parent::copy_block_to_block(d, dstOffset, s, srcOffset, count, err);

    ENSURES(dst->get_state() == dstState);
    ENSURES(src->get_state() == srcState);

    return ret;
}

}}}

#endif // USE_DBC
