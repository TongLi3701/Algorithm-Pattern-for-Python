# 链表

## 基本技能

链表相关的核心点

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 常见题型

### [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)

> 给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        
        if head is None:
            return head
        
        current = head
        
        while current.next is not None:
            if current.next.val == current.val:
                current.next = current.next.next
            else:
                current = current.next
        
        return head
```

另外, 可以使用双指针, 更加好理解一些:
```Python
class Solution:
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head: return head
        
        pre, cur = head, head.next
        
        while cur:
            if pre.val == cur.val:
                pre.next = cur.next
            else:
                pre = pre.next
            
            cur = cur.next
        
        return head
```

时间复杂度: O(n)

空间复杂度: O(1)

### [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)

> 给定一个排序链表，删除所有含有重复数字的节点，只保留原始链表中没有重复出现的数字。


方法1: 递归
链表和树的问题, 一般都可以利用递归和迭代两种写法.

1.1 递归函数的定义:
本题目中直接使用`deleteDuplicates(head)`, 用于删除以`head`作为开头的有序链表中, 值出现重复的节点

1.2 递归终止条件:
* `head` 为空, 直接返回head
* `head.next` 为空, 那么说明链表中只有一个节点, 也没有出现重复的节点, 直接返回`head`

1.3 递归调用:
* 若`head.val != head.next.val`, 说明头结点不等于下一个节点的值, 所以当前`head`节点必须保留; 接下来我们需要对`head.next`进行递归, `head.next = self.deleteDuplicates(head.next)`
* 如果`head.val == head.next.val`, 说明头结点等于下一个节点的值, 那么当前头结点则必须要删除. 需要用`move` 指针一直向后遍历寻找与`head.val`不等的节点. 此时`move` 之前的节点都不保留了, 因此返回`deleteDuplicates(move)`;

1.4 
题目让我们返回删除了值重复的节点后剩余的链表，结合上面两种递归调用的情况。

如果 `head.val != head.next.val`, 头结点需要保留，因此返回的是 `head`;
如果 `head.val == head.next.val`, 头结点需要删除，需要返回的是`deleteDuplicates(move)`;

```Python
class Solution(object):
    def deleteDuplicates(self, head):
        if not head or not head.next:
            return head
        if head.val != head.next.val:
            head.next = self.deleteDuplicates(head.next)
        else:
            move = head.next
            while move and head.val == move.val:
                move = move.next
            return self.deleteDuplicates(move)
        return head
```

### [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)

> 反转一个单链表。

- 思路：将当前结点放置到头结点

方法1: 利用stack的特性可以直接反转链表的value, 但是会占用多余的空间
```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        stack = []
        
        while head:
            stack.append(head.val)
            head = head.next
        
        dummy = ListNode(-1)
        a = dummy
        while stack:
            val = stack.pop()
            a.next = ListNode(val)
            a = a.next
        
        return dummy.next
```

方法2: 利用双指针, `pre` and `cur`, 注意要先保存`cur.next` 防止后面元素的丢失.
```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        pre, cur = None, head
        
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        
        return pre
```
方法3: recursion
```Python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
    	if not head or not head.next: return head
        
        last = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        
        return last
```

### [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)

> 反转从位置  *m*  到  *n*  的链表。请使用一趟扫描完成反转。

- 思路：先找到 m 处, 再反转 n - m 次即可

注意: 
1. 通常情况下, 我们需要先定义一个dummy head, 然后让dummy head指向haed


方法1: 头插法, 将后面的元素一个一个往前插入
步骤:
 - 首先定义2个指针, 分别称为 pre 和 cur
 - 根据 left 和 right 来确定 pre 和 cur 的位置
 - 将 pre 移动到第一个要翻转的节点前面, 将 cur 移动到第一个要翻转的节点的位置上, 以 m=2, n=4 为例
 - 将 cur 后面的元素删除, 然后添加到 pre 的后面
 - 然后不断重复

```Python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        dummy = ListNode(0)
        dummy.next = head
        
        pre = dummy
        cur = head
        
        for i in range(left - 1):
            pre = pre.next
            cur = cur.next
        
        for i in range(right - left):
            removed = cur.next
            cur.next = cur.next.next # delete the removed node
            
            removed.next = pre.next
            pre.next = removed
        
        return dummy.next

```
在链表中, remove node 的操作就是 next.next, 这样可以直接跳过一个 node, 但是在remove之前,一定要先把这个node记录下来,以防丢失


```Python
class Solution:
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        stack = []
        dummy = ListNode(0)
        dummy.next = head
        
        cur = dummy
        
        for i in range(left - 1):
            cur = cur.next
        
        pre = cur
        cur = cur.next
        
        for i in range(right - left + 1):
            stack.append(cur.val)
            cur = cur.next
        
        while stack:
            pre.next = ListNode(stack.pop())
            pre = pre.next
        
        pre.next = cur
        
        return dummy.next
```
利用stack来实现中间部分内容的翻转, 一定要注意cur到底到了哪里, 以及index的变化.


### [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

> 将两个升序链表合并为一个新的升序链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

方法1: 利用双指针, 遍历两个list

```Python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        
        tail = dummy = ListNode()
        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                tail.next = l2
                l2 = l2.next
            else:
                tail.next = l1
                l1 = l1.next
            tail = tail.next
                
        if l1 is None:
            tail.next = l2
        else:
            tail.next = l1

        return dummy.next
```

方法2: recursion
```Python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1: return l2  # 终止条件，直到两个链表都空
        if not l2: return l1
        if l1.val <= l2.val:  # 递归调用
            l1.next = self.mergeTwoLists(l1.next,l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1,l2.next)
            return l2
```


### [partition-list](https://leetcode-cn.com/problems/partition-list/)

> 给定一个链表和一个特定值 x，对链表进行分隔，使得所有小于  *x*  的节点都在大于或等于  *x*  的节点之前。

- 思路：将大于 x 的节点，放到另外一个链表，最后连接这两个链表

```go
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        
        p = l = ListNode()
        q = s = ListNode(next=head)
        
        while q.next is not None:
            if q.next.val < x:
                q = q.next
            else:
                p.next = q.next
                q.next = q.next.next
                p = p.next
        
        p.next = None
        q.next = l.next
        
	return s.next
```

哑巴节点使用场景

> 当头节点不确定的时候，使用哑巴节点

### [sort-list](https://leetcode-cn.com/problems/sort-list/)

> 在  *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。

- 思路：归并排序，slow-fast找中点

```Python
class Solution:
    
    def _merge(self, l1, l2):
        tail = l_merge = ListNode()
        
        while l1 is not None and l2 is not None:
            if l1.val > l2.val:
                tail.next = l2
                l2 = l2.next
            else:
                tail.next = l1
                l1 = l1.next
            tail = tail.next

        if l1 is not None:
            tail.next = l1
        else:
            tail.next = l2
        
        return l_merge.next
    
    def _findmid(self, head):
        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
        
        return slow
    
    def sortList(self, head: ListNode) -> ListNode:
        if head is None or head.next is None:
            return head
        
        mid = self._findmid(head)
        tail = mid.next
        mid.next = None # break from middle
        
        return self._merge(self.sortList(head), self.sortList(tail))
```

注意点

- 快慢指针 判断 fast 及 fast.Next 是否为 nil 值
- 递归 mergeSort 需要断开中间节点
- 递归返回条件为 head 为 nil 或者 head.Next 为 nil

### [reorder-list](https://leetcode-cn.com/problems/reorder-list/)

> 给定一个单链表  *L*：*L*→*L*→…→*L\_\_n*→*L*
> 将其重新排列后变为： *L*→*L\_\_n*→*L*→*L\_\_n*→*L*→*L\_\_n*→…

- 思路：找到中点断开，翻转后面部分，然后合并前后两个链表

```Python
class Solution:
    
    def reverseList(self, head: ListNode) -> ListNode:
        
        prev, curr = None, head
        
        while curr is not None:
            curr.next, prev, curr = prev, curr, curr.next
            
        return prev
    
    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """
        if head is None or head.next is None or head.next.next is None:
            return
        
        slow, fast = head, head.next
        while fast is not None and fast.next is not None:
            fast = fast.next.next
            slow = slow.next
        
        h, m = head, slow.next
        slow.next = None
        
        m = self.reverseList(m)
        
        while h is not None and m is not None:
            p = m.next
            m.next = h.next
            h.next = m
            h = h.next.next
            m = p
            
        return
```

### [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)

> 给定一个链表，判断链表中是否有环。

- 思路1：Hash Table 记录所有结点判断重复，空间复杂度 O(n) 非最优，时间复杂度 O(n) 但必然需要 n 次循环
- 思路2：快慢指针，快慢指针相同则有环，证明：如果有环每走一步快慢指针距离会减 1，空间复杂度 O(1) 最优，时间复杂度 O(n) 但循环次数小于等于 n
  ![fast_slow_linked_list](https://img.fuiboom.com/img/fast_slow_linked_list.png)

```Python
class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        
        slow = fast = head
        
        while fast is not None and fast.next is not None:
            slow = slow.next
	    fast = fast.next.next
            if fast == slow:
                return True
        
        return False
```

### [linked-list-cycle-ii](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

> 给定一个链表，返回链表开始入环的第一个节点。  如果链表无环，则返回  `null`。

- 思路：快慢指针，快慢相遇之后，慢指针回到头，快慢指针步调一致一起移动，相遇点即为入环点。

![cycled_linked_list](https://img.fuiboom.com/img/cycled_linked_list.png)

```Python
class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        
        slow = fast = head
        
        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next
            
            if slow == fast:
                slow = head
                while fast != slow:
                    fast = fast.next
                    slow = slow.next
                return slow

        return None
```

坑点

- 指针比较时直接比较对象，不要用值比较，链表中有可能存在重复值情况
- 第一次相交后，快指针需要从下一个节点开始和头指针一起匀速移动


注意，此题中使用 slow = fast = head 是为了保证最后找环起始点时移动步数相同，但是作为找中点使用时**一般用 fast=head.Next 较多**，因为这样可以知道中点的上一个节点，可以用来删除等操作。

- fast 如果初始化为 head.Next 则中点在 slow.Next
- fast 初始化为 head,则中点在 slow

### [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)

> 请判断一个链表是否为回文链表。

- 思路：O(1) 空间复杂度的解法需要破坏原链表（找中点 -> 反转后半个list -> 判断回文），在实际应用中往往还需要复原（后半个list再反转一次后拼接），操作比较复杂，这里给出更工程化的做法

```Python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        
        s = []
        slow = fast = head
        while fast is not None and fast.next is not None:
            s.append(slow.val)
            slow = slow.next
            fast = fast.next.next
        
        if fast is not None:
            slow = slow.next
        
        while len(s) > 0:
            if slow.val != s.pop():
                return False
            slow = slow.next
            
        return True
```

### [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)

> 给定一个链表，每个节点包含一个额外增加的随机指针，该指针可以指向链表中的任何节点或空节点。
> 要求返回这个链表的 深拷贝。

- 思路1：hash table 存储 random 指针的连接关系

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if head is None:
            return None
        
        parent = collections.defaultdict(list)
        
        out = Node(0)
        o, n = head, out
        while o is not None:
            n.next = Node(o.val)
            n = n.next
            if o.random is not None:
                parent[o.random].append(n)
            o = o.next
            
        o, n = head, out.next
        while o is not None:
            if o in parent:
                for p in parent[o]:
                    p.random = n
            o = o.next
            n = n.next
        
        return out.next
```

- 思路2：复制结点跟在原结点后面，间接维护连接关系，优化空间复杂度，建立好新 list 的 random 链接后分离

```Python
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        
        if head is None:
            return None
        
        p = head
        while p is not None:
            p.next = Node(p.val, p.next)
            p = p.next.next
        
        p = head
        while p is not None:
            if p.random is not None:
                p.next.random = p.random.next
            p = p.next.next
        
        new = head.next
        o, n = head, new
        while n.next is not None:
            o.next = n.next
            n.next = n.next.next
            o = o.next
            n = n.next
        o.next = None
        
        return new
```

## 总结

链表必须要掌握的一些点，通过下面练习题，基本大部分的链表类的题目都是手到擒来~

- null/nil 异常处理
- dummy node 哑巴节点
- 快慢指针
- 插入一个节点到排序链表
- 从一个链表中移除一个节点
- 翻转链表
- 合并两个链表
- 找到链表的中间节点

## 练习

- [ ] [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
- [ ] [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
- [ ] [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)
- [ ] [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
- [ ] [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
- [ ] [partition-list](https://leetcode-cn.com/problems/partition-list/)
- [ ] [sort-list](https://leetcode-cn.com/problems/sort-list/)
- [ ] [reorder-list](https://leetcode-cn.com/problems/reorder-list/)
- [ ] [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
- [ ] [linked-list-cycle-ii](https://leetcode-cn.com/problems/https://leetcode-cn.com/problems/linked-list-cycle-ii/)
- [ ] [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)
- [ ] [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
