#include <algorithm>

#include "mapping.h"

namespace __impl { namespace dsm {

mapping::range_block
mapping::get_blocks_in_range(size_t offset, size_t count)
{
    ASSERTION(offset < size_);
    ASSERTION(offset + count <= size_);

    size_t tmp = 0;

    list_block::iterator first = blocks_.begin();
    
    while (true) {
        tmp += (*first)->get_size();
        if (tmp >= offset) break;
        first++;
    }

    list_block::iterator last = first;
    if (last != blocks_.end()) last++;
    while (tmp < offset + count) {
        tmp += (*last)->get_size();
        // We increment before the check because we return an open range
        last++;
    }

    return range_block(first, last);
}

template <bool forward>
void
mapping::shift_blocks(size_t offset)
{
    TRACE(LOCAL, FMT_ID2" Shifting blocks by " FMT_SIZE" bytes", get_print_id2(), offset);

    for (coherence::block_ptr b : blocks_) {
        b->shift(this, size_);
    }
}

mapping::cursor_block
mapping::get_first_block(size_t offset)
{
    list_block::iterator it;

    size_t off = 0;
    for (list_block::iterator it  = blocks_.begin();
                              it != blocks_.end();
                            ++it) {
        size_t blockSize = (*it)->get_size();

        if ((offset >= off) &&
            (offset <  off + blockSize)) {
            return cursor_block(it, off, offset - off);
        }

        off += blockSize;
    }

    return cursor_block(blocks_.end(), 0, 0);
}

mapping::cursor_block
mapping::split_block(cursor_block cursor, size_t offset)
{
    TRACE(LOCAL, FMT_ID2" Splitting block at offset " FMT_SIZE, get_print_id2(), offset);

    // coherence::block_ptr blockNew =
    		cursor.get_block()->split(offset);

    // list_block::iterator it = blocks_.insert(++cursor.get_iterator(), blockNew);

    //return cursor_block(it, cursor.get_offset_block(), cursor.get_offset_local());
    return get_first_block(offset);
}

#define H(a) typeof(a)

error
mapping::prepend(coherence::block_ptr b)
{
    TRACE(LOCAL, FMT_ID2" Prepending " FMT_ID2, get_print_id2(), b->get_print_id2());

    error ret = error::DSM_SUCCESS;

    // TODO: figure out why the compiler complaints without the cast
    if (b->get_size() <= size_t(addr_.get_offset())) {
        shift_blocks(b->get_size());

        blocks_.push_front(b);
        b->register_mapping(this, 0);
        addr_ -= b->get_size();
        size_ += b->get_size();
    } else {
        ret = error::DSM_ERROR_INVALID_VALUE;
    }

    return ret;
}

error
mapping::append(coherence::block_ptr b)
{
    TRACE(LOCAL, FMT_ID2" Appending " FMT_ID2,
                 get_print_id2(), b->get_print_id2());

    error ret = error::DSM_SUCCESS;

    if (b->get_size() > 0) {
        blocks_.push_back(b);
        b->register_mapping(this, size_);

        size_ += b->get_size();
    } else {
        ret = error::DSM_ERROR_INVALID_VALUE;
    }

    return ret;
}

error
mapping::merge(coherence::block_ptr b, coherence::block_ptr bNew)
{
    TRACE(LOCAL, FMT_ID2" Swapping " FMT_ID2" with " FMT_ID2,
                 get_print_id2(), b->get_print_id2(), bNew->get_print_id2());

    ASSERTION(b != bNew);
    ASSERTION(std::find(blocks_.begin(), blocks_.end(), b) != blocks_.end(),
              "Block does not belong to this mapping");
    ASSERTION(std::find(blocks_.begin(), blocks_.end(), bNew) == blocks_.end(),
              "Block already found in mapping");

    error ret = bNew->transfer_mappings(std::move(*b));

    if (ret == error::DSM_SUCCESS) {
        list_block::iterator it = std::find(blocks_.begin(), blocks_.end(), b);
        *it = bNew;
    }

    return ret;
}

error
mapping::move_block(mapping &dst, mapping &src, coherence::block_ptr b)
{
    error err;
    // Remove references to the current mapping
    err = b->unregister_mapping(src);
    if (err != error::DSM_SUCCESS) return err;
    // Update mapping size
    src.size_ -= b->get_size();
    // Append block to the new mapping
    err = dst.append(b);

    return err;
}

error
mapping::append(mapping &&m)
{
    TRACE(LOCAL, FMT_ID2" Appending " FMT_ID2,
                 get_print_id2(), m.get_print_id2());

    // TODO: figure out why the compiler complaints without the cast
    if ((size_t(m.addr_.get_offset()) <  (addr_.get_offset() + size_)) ||
        (&m.addr_.get_view() !=  &addr_.get_view())) {
        return error::DSM_ERROR_INVALID_VALUE;
    }

    error ret = error::DSM_SUCCESS;

    // Add a new block between the mappings if needed
    if (m.get_bounds().start > get_bounds().end) {
        coherence::block_ptr b = factory_block::create(m.get_bounds().start - get_bounds().end);

        blocks_.push_back(b);

        b->register_mapping(this, size_);

        size_ += b->get_size(); 
    }

    // Insert the rest of blocks into the m
    for (coherence::block_ptr b : m.blocks_) {
        blocks_.push_back(b);

        b->register_mapping(this, size_);
        b->unregister_mapping(m);

        size_ += b->get_size(); 
    }

    m.blocks_.clear();

    return ret;
}

error
mapping::resize(size_t pre, size_t post)
{
    error ret = error::DSM_SUCCESS;

    TRACE(LOCAL, FMT_ID2" Resizing [" FMT_SIZE", " FMT_SIZE"]", get_print_id2(), pre, post);

    if (pre > 0) {
        coherence::block_ptr b = factory_block::create(pre);
        ret = prepend(b);
    }

    if (post > 0 && ret == error::DSM_SUCCESS) {
        coherence::block_ptr b = factory_block::create(post);
        ret = append(b);
    }

    return ret;
}

mapping::pair_mapping
mapping::split(size_t offset, size_t count, error &err)
{
    TRACE(LOCAL, FMT_ID2" Splitting at " FMT_SIZE ", " FMT_SIZE "bytes", get_print_id2(), offset, count);

    mapping_ptr mNew1 = nullptr, mNew2 = nullptr;

    mNew1 = new mapping(addr_ + offset, prot_, flags_);

    // Deal with the first block
    cursor_block cursor = get_first_block(offset);

    if (cursor.get_offset_local() != 0) {
    	// Does not start on block beginning, split the block
    	split_block(cursor, cursor.get_offset_local());
    }

    // Deal with the last block
    cursor = get_first_block(offset + count);

    if (cursor.get_offset_local() != 0) {
        // Does not end on block ending, split the block
        split_block(cursor, cursor.get_offset_local());
    }

    cursor = get_first_block(offset);

    ASSERTION(cursor.get_offset_local() == 0);

    size_t off = 0;

    while (off < count) {
        coherence::block_ptr b = cursor.get_block();

        mapping::move_block(*mNew1, *this, b);

        if (err != error::DSM_SUCCESS) {
            return pair_mapping(nullptr, nullptr);
        }

        ASSERTION(off + b->get_size() <= count);
        off += b->get_size();

        // Remove the block and advance the iterator
        auto it = cursor.get_iterator();
        cursor.advance_block();
        blocks_.erase(it);
    }

    if ((offset > 0) &&
        (offset + count < size_)) {
    	// Create another mapping if needed
        mNew2 = new mapping(addr_ + offset + count, prot_, flags_);

        // Move the remaining blocks
        while (cursor.get_iterator() != blocks_.end()) {
        	coherence::block_ptr b = cursor.get_block();

        	mapping::move_block(*mNew2, *this, b);

        	if (err != error::DSM_SUCCESS) {
        		return pair_mapping(nullptr, nullptr);
        	}

        	// Remove the block and advance the iterator
        	auto it = cursor.get_iterator();
        	cursor.advance_block();
        	blocks_.erase(it);
        }
    }

    return pair_mapping(mNew1, mNew2);
}

mapping::mapping(hal::ptr addr, GmacProtection prot, int flags) :
    addr_(addr),
    size_(0),
    prot_(prot),
    flags_(flags)
{
    TRACE(LOCAL, FMT_ID2" Creating", get_print_id2());
}

mapping::mapping(mapping &&m) :
    addr_(m.addr_),
    size_(m.size_),
    prot_(m.prot_),
    blocks_(m.blocks_)
{
    TRACE(LOCAL, FMT_ID2" Creating", get_print_id2());

    size_t size = 0;
    for (coherence::block_ptr b : blocks_) {
        b->unregister_mapping(m);
        b->register_mapping(this, size);

        size += b->get_size();
    }

    m.blocks_.clear();
}

mapping::~mapping()
{
    TRACE(LOCAL, FMT_ID2" Deleting", get_print_id2());

    for (coherence::block_ptr b : blocks_) {
        b->unregister_mapping(*this);
    }

    blocks_.clear();
}

error
mapping::acquire(size_t offset, size_t count, GmacProtection prot)
{
    TRACE(LOCAL, FMT_ID2" Acquire " FMT_SIZE":" FMT_SIZE,
                 get_print_id2(), offset, count);

    CHECK(prot_is_readable(prot_) || !prot_is_readable(prot), error::DSM_ERROR_INVALID_PROT);
    CHECK(prot_is_writable(prot_) || !prot_is_writable(prot), error::DSM_ERROR_INVALID_PROT);

    error ret = error::DSM_SUCCESS;

    range_block range = get_blocks_in_range(offset, count);

    for (coherence::block_ptr b : range) {
        ret = b->acquire(this, prot);
        if (ret != error::DSM_SUCCESS) break; 
    }

    return ret;
}

error
mapping::release(size_t offset, size_t count)
{
    TRACE(LOCAL, FMT_ID2" Release " FMT_SIZE":" FMT_SIZE,
                 get_print_id2(), offset, count);

    error err = error::DSM_SUCCESS;

    range_block range = get_blocks_in_range(offset, count);

    for (coherence::block_ptr b : range) {
        err = b->release(this);
        if (err != error::DSM_SUCCESS) break;
    }

    return err;
}

error
mapping::handle_fault(size_t offLocal, bool isWrite)
{
    CHECK(offLocal < size_, error::DSM_ERROR_INVALID_VALUE);

    cursor_block cursor = get_first_block(offLocal);
    ASSERTION(cursor.get_iterator() != blocks_.end());

    error err;

    if (isWrite) {
        err = cursor.get_block()->handle_fault<true>(this, cursor.get_offset_local());
    } else {
        err = cursor.get_block()->handle_fault<false>(this, cursor.get_offset_local());
    }

    return err;
}

error
mapping::link(size_t offDst, mapping_ptr mDst,
              size_t offSrc, mapping_ptr mSrc, size_t count, int flags)
{
    TRACE(STATIC(mapping), "dsm::mapping linking " FMT_ID2" and " FMT_ID2,
                           mDst->get_print_id2(),
                           mSrc->get_print_id2());

    CHECK(mDst != mSrc, error::DSM_ERROR_INVALID_VALUE);

    CHECK(count > 0, error::DSM_ERROR_INVALID_VALUE);

    CHECK(offDst + count <= mDst->get_bounds().get_size(), error::DSM_ERROR_INVALID_VALUE);
    CHECK(offSrc + count <= mSrc->get_bounds().get_size(), error::DSM_ERROR_INVALID_VALUE);

    error ret = error::DSM_SUCCESS;

    cursor_block cursorDst = mDst->get_first_block(offDst);
    cursor_block cursorSrc = mSrc->get_first_block(offSrc);

    ASSERTION(cursorDst.get_iterator() != mDst->blocks_.end());
    ASSERTION(cursorSrc.get_iterator() != mSrc->blocks_.end());

    // If we are mapping in the middle of a block...
    if (cursorDst.get_offset_local() > 0) {
        // ... split the block and move to the newly created block
        mDst->split_block(cursorDst, cursorDst.get_offset_local());
        cursorDst.advance_block();
    }
    // If we are mapping in the middle of a block...
    if (cursorSrc.get_offset_local() > 0) {
        // ... split the block and move to the newly created block
        mSrc->split_block(cursorSrc, cursorSrc.get_offset_local());
        cursorSrc.advance_block();
    }

    do {
        size_t bytesSubmapping = 0;

        // Both cursors MUST point at the beginning of a block
        ASSERTION(cursorDst.get_offset_local() == 0);
        ASSERTION(cursorSrc.get_offset_local() == 0);

        if ((cursorDst.get_block()->get_size() >= count) &&
            (cursorSrc.get_block()->get_size() >= count)) {
            // If the blocks are bigger than the mapping, split them
            if (cursorDst.get_block()->get_size() > count) {
                mDst->split_block(cursorDst, count);
            }
            if (cursorSrc.get_block()->get_size() > count) {
                mSrc->split_block(cursorSrc, count);
            }

            // Merge block owners into a single block
            ret = mDst->merge(cursorDst.get_block(), cursorSrc.get_block());
            if (ret != error::DSM_SUCCESS) {
                break;
            }

            // We do not advance the iterator since we are done
            bytesSubmapping = count;
        } else {
            if ((cursorDst.get_block()->get_size() < count) &&
                (cursorSrc.get_block()->get_size() < count)) {
                // If both blocks are smaller than the remaining size of the mapping
                if (cursorDst.get_block()->get_size() ==
                    cursorSrc.get_block()->get_size()) {
                    // Merge block owners into a single block
                    cursorDst.get_block()->transfer_mappings(std::move(*cursorSrc.get_block()));

                    bytesSubmapping = cursorDst.get_block()->get_size();

                    // If both blocks are equally sized, advance both cursors
                    cursorDst.advance_block();
                    cursorSrc.advance_block();
                } else if (cursorDst.get_block()->get_size() <
                           cursorSrc.get_block()->get_size()) {
                    // Split src
                    mSrc->split_block(cursorSrc, cursorDst.get_block()->get_size());
                    // Merge block owners into a single block
                    cursorDst.get_block()->transfer_mappings(std::move(*cursorSrc.get_block()));
                    // Move to next block in dst
                    size_t remainder = cursorDst.advance_block();
                    // Move to next block in src
                    cursorSrc.advance_block();

                    bytesSubmapping = remainder;
                } else if (cursorSrc.get_block()->get_size() <
                           cursorDst.get_block()->get_size()) {
                    // Split dst
                    mDst->split_block(cursorDst, cursorSrc.get_block()->get_size());
                    // Merge block owners into a single block
                    cursorDst.get_block()->transfer_mappings(std::move(*cursorSrc.get_block()));
                    // Move to next block in src
                    size_t remainder = cursorSrc.advance_block();
                    // Move to next block in dst
                    cursorDst.advance_block();

                    bytesSubmapping = remainder;
                }
            } else if (cursorDst.get_block()->get_size() < count) {
                // If dst block is smaller than the remaining size of the mapping

                // Split src
                mSrc->split_block(cursorSrc, cursorDst.get_block()->get_size());
                // Merge block owners into a single block
                cursorDst.get_block()->transfer_mappings(std::move(*cursorSrc.get_block()));
                // Move to next block in dst
                size_t remainder = cursorDst.advance_block();
                // Move to next block in src
                cursorSrc.advance_block();

                bytesSubmapping = remainder;
            } else /*  cursorSrc.get_block()->get_size() < count */ {
                // If src block is smaller than the remaining size of the mapping

                // Split src
                mDst->split_block(cursorDst, cursorSrc.get_block()->get_size());
                // Merge block owners into a single block
                cursorDst.get_block()->transfer_mappings(std::move(*cursorSrc.get_block()));
                // Move to next block in src
                size_t remainder = cursorSrc.advance_block();
                // Move to next block in dst
                cursorDst.advance_block();

                bytesSubmapping = remainder;
            }
        }

        count -= bytesSubmapping;
    } while (count > 0);

    return ret;
}

}}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
