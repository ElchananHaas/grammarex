struct QueueItemHeader {
    //The machine type
    machine: usize,
    //The current state within the machine
    cur_state: usize,
    //An index to a bitset for tracking the active states
    active_state_idx: usize, 
}

impl QueueItemHeader {
    fn get_machine(&self) -> usize {
        self.machine
    }
}

struct MultiQueue<'a> {
    var_sizes: &'a Vec<usize>,
    data: Vec<u8>,
}

//This is a queue for storing multiple types of items of varying sizes. 
//Each item starts with a header which can be used to determine its full size.
impl<'a> MultiQueue<'a> {

    fn item_size(&self, item: &QueueItemHeader) -> usize {
        size_of::<QueueItemHeader>() + self.var_sizes[item.get_machine()]
    }
    pub fn push(&mut self, item: &QueueItemHeader) {
        let current_data_len = self.data.len();
        let item_num_bytes = self.item_size(item);
        for _ in 0..item_num_bytes {
            self.data.push(0);
        }
        let current_end_ptr = &mut self.data[current_data_len] as *mut u8;
        let item_ptr = item as *const QueueItemHeader as *const u8;
        // SAFETY: There is an invariant that QueueItemHeaders are followed by 
        // self.var_sizes[item.get_machine()] in memory. This trailing data is Copy
        // To uphold this invariant QueueItemHeaders have private fields and MultiQueue has an unsafe constructor. 
        // This method upholds the invariant in the queue too.
        unsafe {
            std::ptr::copy_nonoverlapping(item_ptr, current_end_ptr, item_num_bytes);
        }
    }
}

struct MultiQueueIterator<'a> {
    queue: MultiQueue<'a>,
    offset: usize
}

impl<'a> Iterator for MultiQueueIterator<'a> {
    type Item = &'a QueueItemHeader;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset < self.queue.data.len() {
            let data_ptr = &self.queue.data[self.offset] as *const u8 as *const QueueItemHeader;
            let item: &'a QueueItemHeader = unsafe {
                 &*data_ptr
            };
            let item_len = self.queue.item_size(item);
            self.offset += item_len;
            Some(item)
        } else {
            None
        }
    }
}

impl<'a> IntoIterator for MultiQueue<'a> {
    type Item = &'a QueueItemHeader;

    type IntoIter = MultiQueueIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        MultiQueueIterator {
            queue: self,
            offset: 0
        }
    }
}