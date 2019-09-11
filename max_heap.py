#coding=utf-8

class MaxHeap(object):
    # 创建MaxHeap类
    def __init__(self, max_size, fn):
        self.max_size = max_size
        self.fn = fn
        self.items = [None] * max_size
        self.size = 0
    # 打印对象中具体的属性值
    def __str__(self):
        item_values = str([self.fn(self.items[i]) for i in range(self.size)])
        return ("Size: %d\nMax size: %d\nItem_values: %s\n" % (self.size, self.max_size, item_values))
    # 检查大顶堆是否已满
    @property
    def full(self):
        return self.size == self.max_size
    # 添加元素
    @property
    def add(self, item):
        if self.full:
            if self.fn(item) < self.value(0):
                self.items[0] = item
                self.shift_down(0)
        else:
            self.items[self.size] = item
            self.size += 1
            self.shift_up(self.size - 1)
    # 推出顶部元素
    def pop(self):
        assert self.size > 0, "Cannot pop item! The MaxHeap is empty!"
        ret = self.items[0]
        self.items[0] = self.items[self.size - 1]
        self.items[self.size - 1] = None
        self.size -= 1
        self.shit_down(0)
        return ret
    # 元素上浮
    def shift_up(self, idx):
        assert idx < self.size, "The parameter idx must be less than heap's size!"
        parent = (idx - 1) // 2
        while parent >= 0 and self.value(parent) < self.value(idx):
            self.items[parent], self.items[idx] = self.items[idx], self.items[parent]
            idx = parent
            parent = (idx - 1) // 2

    # 元素下沉
    def shift_down(self, idx):
        child = (idx + 1) * 2 - 1
        while child < self.size:
            if child + 1 < self.size and self.value(child + 1) > self.value(child):
                child += 1
            if self.value(idx) < self.value(child):
                self.items[idx], self.items[child] = self.items[child], self.items[idx]
                idx = child
                child = (idx + 1) * 2 - 1
            else:
                break
    # 检查有效性
    def is_valid(self):
        ret = []
        for i in range(1, self.size):
            parent = (i - 1) // 2
            ret.append(self.value(parent) >= self.value(i))
        # all()函数用于判定可迭代参数iterable中的所有元素是否都为TRUE，如果是返回True，否则返回False
        return all(ret)


