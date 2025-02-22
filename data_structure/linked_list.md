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

- 思路：将大于等于 x 的 node 构建一个新的 list, 将小于 x 的 node 构建一个 list,然后将两者拼接起来

```Python
class Solution:
    def partition(self, head: ListNode, x: int) -> ListNode:
        before_head, after_head = ListNode(), ListNode()
        before, after = before_head, after_head
        
        while head:
            if head.val < x:
                before.next = ListNode(head.val)
                before = before.next
                head = head.next
            else:
                after.next = ListNode(head.val)
                after = after.next
                head = head.next
        
        before.next = after_head.next
        
        return before_head.next
```

哑巴节点使用场景

> 当头节点不确定的时候，使用哑巴节点

### [sort-list](https://leetcode-cn.com/problems/sort-list/)

> 在  *O*(*n* log *n*) 时间复杂度和常数级空间复杂度下，对链表进行排序。


1. 方法1: 先保存节点的value, 然后排序, 最后从新生成链表
```Python
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return head
        dummy = ListNode(head.val)

        node_list = []
        while head:
            node_list.append(head.val)
            head = head.next
        
        node_list = sorted(node_list)
        cur = dummy
        for val in node_list:
            cur.next = ListNode(val)
            cur = cur.next
        
        return dummy.next
```

方法2: 归并排序，利用快慢指针找到链表的中点

```Python
class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        if not head or not head.next: return head # termination.
        # cut the LinkedList at the mid index.
        slow, fast = head, head.next
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
        mid, slow.next = slow.next, None # save and cut.
        # recursive for cutting.
        left, right = self.sortList(head), self.sortList(mid)
        # merge `left` and `right` linked list and return it.
        h = res = ListNode(0)
        while left and right:
            if left.val < right.val: h.next, left = left, left.next
            else: h.next, right = right, right.next
            h = h.next
        h.next = left if left else right
        return res.next
```
注意点: 
1. slow fast 指针一定位置是否正确
2. 找到中间的时候一定要断开
3. 记得保存头结点

### [reorder-list](https://leetcode-cn.com/problems/reorder-list/)

> 给定一个单链表  *L*：*L*→*L*→…→*L\_\_n*→*L*
> 将其重新排列后变为： *L*→*L\_\_n*→*L*→*L\_\_n*→*L*→*L\_\_n*→…

- 思路：找到中点断开，翻转后面部分，然后合并前后两个链表

```Python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return
        
        mid = self.middleNode(head)
        l1 = head
        l2 = mid.next
        mid.next = None
        l2 = self.reverseList(l2)
        self.mergeList(l1, l2)
    
    def middleNode(self, head: ListNode) -> ListNode:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
    
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr:
            nextTemp = curr.next
            curr.next = prev
            prev = curr
            curr = nextTemp
        return prev

    def mergeList(self, l1: ListNode, l2: ListNode):
        while l1 and l2:
            l1_tmp = l1.next
            l2_tmp = l2.next

            l1.next = l2
            l1 = l1_tmp

            l2.next = l1
            l2 = l2_tmp
```

方法2: 利用线性表, 存储该链表的值, 然后利用线性表下标访问的特点, 直接按照顺序访问指点链表, 重建该链表, 但是缺点是比较占用空间. 空间复杂度是该链表的节点数.
```Python
class Solution:
    def reorderList(self, head: ListNode) -> None:
        if not head:
            return
        
        vec = list()
        node = head
        while node:
            vec.append(node)
            node = node.next
        
        i, j = 0, len(vec) - 1
        while i < j:
            vec[i].next = vec[j]
            i += 1
            if i == j:
                break
            vec[j].next = vec[i]
            j -= 1
        
        vec[i].next = None
```

### [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)

> 给定一个链表，判断链表中是否有环。

- 思路1：Hash Table 记录所有结点判断重复，空间复杂度 O(n) 非最优，时间复杂度 O(n) 但必然需要 n 次循环
- 思路2：快慢指针，快慢指针相同则有环，证明：如果有环每走一步快慢指针距离会减 1，空间复杂度 O(1) 最优，时间复杂度 O(n) 但循环次数小于等于 n

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

<!-- ![cycled_linked_list](https://img.fuiboom.com/img/cycled_linked_list.png) -->

![image](https://user-images.githubusercontent.com/52250342/148849412-d602dc4f-5164-4163-bf61-2b81672a310b.png)

* slow * 2 = fast
* slow = a + b
* fast = a + b + c + b = a + 2*b + c
* (a + b)*2 = a + 2*b + c
* a = c

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

方法1:
储存列表的值, 然后进行比较

```Python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        cur = head
        s = []
        while cur:
            s.append(cur.val)
            cur = cur.next 
        
        for i in range(len(s) // 2):
            if s[i] == s[len(s) - 1 - i]:
                continue 
            else:
                return False
        
        return True
```



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

- 思路1：dictionary 存储 random 指针的连接关系

```Python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return head
        # save head node
        cur = head
        
        node_dic = dict()
        
        while cur:
            node_dic[cur] = Node(cur.val)
            cur = cur.next
        
        cur = head
        
        while cur:
            if cur.next:
                node_dic[cur].next = node_dic[cur.next]
            if cur.random:
                node_dic[cur].random = node_dic[cur.random]
            
            cur = cur.next
        
        return node_dic[head]
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


双指针在不同情况的处理时, 需求不同:
方式1:
```Python
def middleNode(self, head: ListNode) -> ListNode:
    slow, fast = head, head 
        
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next 
       
    return slow
```
[1, 2, 3, 4, 5]
first = [1, 2]
second = [3, 4, 5]

方式2:
```
def middleNode(self, head: ListNode) -> ListNode:
    slow, fast = head, head.next
        
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next 
        
     return slow.next
```
[1, 2, 3, 4, 5]
first = [1, 2, 3]
second = [4, 5]

注意看清题目到底是什么需求

## 练习

- [x] [remove-duplicates-from-sorted-list](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list/)
- [x] [remove-duplicates-from-sorted-list-ii](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list-ii/)
- [x] [reverse-linked-list](https://leetcode-cn.com/problems/reverse-linked-list/)
- [x] [reverse-linked-list-ii](https://leetcode-cn.com/problems/reverse-linked-list-ii/)
- [x] [merge-two-sorted-lists](https://leetcode-cn.com/problems/merge-two-sorted-lists/)
- [x] [partition-list](https://leetcode-cn.com/problems/partition-list/)
- [x] [sort-list](https://leetcode-cn.com/problems/sort-list/)
- [x] [reorder-list](https://leetcode-cn.com/problems/reorder-list/)
- [x] [linked-list-cycle](https://leetcode-cn.com/problems/linked-list-cycle/)
- [x] [linked-list-cycle-ii](https://leetcode-cn.com/problems/https://leetcode-cn.com/problems/linked-list-cycle-ii/)
- [x] [palindrome-linked-list](https://leetcode-cn.com/problems/palindrome-linked-list/)
- [x] [copy-list-with-random-pointer](https://leetcode-cn.com/problems/copy-list-with-random-pointer/)
