#ifndef GMAC_HAL_DETAIL_VIRT_ASPACE_H_
#define GMAC_HAL_DETAIL_VIRT_ASPACE_H_

#include <list>
#include <queue>

#include "trace/logger.h"

#include "util/attribute.h"
#include "util/lock.h"
#include "util/locked_counter.h"
#include "util/trigger.h"

#include "hal/detail/ptr.h"

namespace __impl { namespace hal { namespace detail {

class _event;
class kernel;
class stream;
class list_event;
typedef util::shared_ptr<_event> event_ptr;

namespace virt {

class aspace;

template <typename T>
class GMAC_LOCAL map_pool :
    std::map<size_t, std::list<T *> >,
    gmac::util::mutex<map_pool<T> > {

    typedef std::list<T *> queue_subset;
    typedef std::map<size_t, queue_subset> Parent;

public:
    map_pool() :
        gmac::util::mutex<map_pool>("map_pool")
    {
    }

    T *pop(size_t size)
    {
        T *ret = NULL;

        this->lock();
        typename Parent::iterator it;
        it = Parent::lower_bound(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;
            if (queue.size() > 0) {
                ret = queue.front();
                queue.pop_front();
            }
        }
        this->unlock();

        return ret;
    }

    bool remove(T *val, size_t size)
    {
        bool ret = false;

        this->lock();
        typename Parent::iterator it;
        it = Parent::lower_bound(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;

            typename queue_subset::iterator it2;
            it2 = std::find(queue.begin(), queue.end(), val);
            if (it2 != queue.end()) {
                queue.erase(it2);
                ret = true;
            }
        }
        this->unlock();

        return ret;
    }

    void push(T *v, size_t size)
    {
    	this->lock();
        typename Parent::iterator it;
        it = Parent::find(size);
        if (it != Parent::end()) {
            queue_subset &queue = it->second;
            queue.push_back(v);
        } else {
            queue_subset queue;
            queue.push_back(v);
            Parent::insert(typename Parent::value_type(size, queue));
        }
        this->unlock();
    }
};

class GMAC_LOCAL buffer {
public:
    enum type {
        ToHost,
        ToDevice,
        DeviceToDevice
    };
private:
    aspace &aspace_;
    size_t size_;

    event_ptr event_;

protected:
    buffer(size_t size, aspace &as);

    type get_type() const;

public:
    virtual host_ptr get_addr() = 0;
    virtual ptr get_device_addr() = 0;
    aspace &get_aspace();
    const aspace &get_aspace() const;
    size_t get_size() const;

    void set_event(event_ptr event);

    gmacError_t wait();
};

class GMAC_LOCAL code_repository
{
public:
    virtual kernel *get_kernel(gmac_kernel_id_t key) = 0;
    virtual kernel *get_kernel(const std::string &name) = 0;
};

class GMAC_LOCAL queue_event :
    std::queue<_event *>,
    gmac::util::spinlock<queue_event> {

    typedef std::queue<_event *> Parent;
    typedef gmac::util::spinlock<queue_event> Lock;

public:
    queue_event();
    _event *pop();
    void push(_event &event);
};

class GMAC_LOCAL aspace :
    public util::unique<aspace, GmacAddressSpaceId>,
    public util::attributes<aspace>,
    public util::observable<aspace, util::event::construct>,
    public util::observable<aspace, util::event::destruct> {
    friend class util::observable<aspace, util::event::destruct>;

private:
    typedef map_pool<void> map_memory;
    typedef map_pool<buffer> map_buffer;

protected:
    typedef util::locked_counter<unsigned, gmac::util::spinlock<aspace> > buffer_counter;

    phys::processing_unit &pu_;
    phys::aspace &pas_;

    static const unsigned &MaxBuffersIn_;
    static const unsigned &MaxBuffersOut_;

    buffer_counter nBuffersIn_;
    buffer_counter nBuffersOut_;

    map_memory mapMemory_;

    map_buffer mapFreeBuffersIn_;
    map_buffer mapFreeBuffersOut_;

    map_buffer mapUsedBuffersIn_;
    map_buffer mapUsedBuffersOut_;

    queue_event queueEvents_;

    host_ptr get_memory(size_t size);
    void put_memory(void *ptr, size_t size);

    buffer *get_input_buffer(size_t size, stream &stream, event_ptr event);
    void put_input_buffer(buffer &buffer);
 
    buffer *get_output_buffer(size_t size, stream &stream, event_ptr event);
    void put_output_buffer(buffer &buffer);

    virtual buffer *alloc_buffer(size_t size, GmacProtection hint, stream &stream, gmacError_t &err) = 0;
    virtual gmacError_t free_buffer(buffer &buffer) = 0;

    aspace(phys::processing_unit &pu, phys::aspace &pas, gmacError_t &err);
    virtual ~aspace();

public:
    phys::processing_unit &get_processing_unit();
    const phys::processing_unit &get_processing_unit() const;

    phys::aspace &get_paspace();
    const phys::aspace &get_paspace() const;

    //virtual ptr alloc(size_t size, gmacError_t &err) = 0;
    //virtual ptr alloc_host_pinned(size_t size, GmacProtection hint, gmacError_t &err) = 0;
    //virtual ptr map(size_t size, gmacError_t &err) = 0;

    virtual ptr map(virt::object &obj, gmacError_t &err) = 0;
    virtual ptr map(virt::object &obj, ptrdiff_t offset, gmacError_t &err) = 0;

    virtual gmacError_t free(ptr acc) = 0;
#if 0
    virtual gmacError_t free_host_pinned(ptr ptr) = 0;
#endif

    virtual bool has_direct_copy(hal::const_ptr ptr1, hal::const_ptr ptr2) = 0;

    virtual code_repository &get_code_repository() = 0;

    virtual event_ptr copy(hal::ptr dst, hal::const_ptr src, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr copy_async(hal::ptr dst, hal::const_ptr src, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;

    virtual event_ptr copy(hal::ptr dst, device_input &input, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr copy(device_output &output, hal::const_ptr src, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr memset(hal::ptr dst, int c, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr copy_async(hal::ptr dst, device_input &input, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr copy_async(device_output &output, hal::const_ptr src, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
    virtual event_ptr memset_async(hal::ptr dst, int c, size_t count, stream &stream, list_event *dependencies, gmacError_t &err) = 0;
};

}

}}}

#endif /* ASPACE_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
