# 栈和队列

## 简介

栈的特点是后入先出

![image.png](https://img.fuiboom.com/img/stack.png)

根据这个特点可以临时保存一些数据，之后用到依次再弹出来，常用于 DFS 深度搜索

队列一般常用于 BFS 广度搜索，类似一层一层的搜索

## Stack 栈

### [min-stack](https://leetcode-cn.com/problems/min-stack/)

> 设计一个支持 push，pop，top 操作，并能在常数时间内检索到最小元素的栈。

- 思路：用两个栈实现或插入元组实现，保证当前最小值在栈顶即可

```Python
class MinStack:

    def __init__(self):
        self.stack = []

    def push(self, x: int) -> None:
        if len(self.stack) > 0:
            self.stack.append((x, min(x, self.stack[-1][1])))
        else:
            self.stack.append((x, x))

    def pop(self) -> int:
        return self.stack.pop()[0]

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]
```

### [evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)

> **波兰表达式计算** > **输入:** ["2", "1", "+", "3", "*"] > **输出:** 9
> **解释:** ((2 + 1) \* 3) = 9

- 思路：通过栈保存原来的元素，遇到表达式弹出运算，再推入结果，重复这个过程

```Python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        
        def comp(or1, op, or2):
            if op == '+':
                return or1 + or2
            
            if op == '-':
                return or1 - or2
            
            if op == '*':
                return or1 * or2
            
            if op == '/':
                abs_result = abs(or1) // abs(or2)
                return abs_result if or1 * or2 > 0 else -abs_result
        
        stack = []
        
        for token in tokens:
            if token in ['+', '-', '*', '/']:
                or2 = stack.pop()
                or1 = stack.pop()
                stack.append(comp(or1, token, or2))
            else:
                stack.append(int(token))
        
        return stack[0]
```

### [decode-string](https://leetcode-cn.com/problems/decode-string/)

> 给定一个经过编码的字符串，返回它解码后的字符串。
> s = "3[a]2[bc]", 返回 "aaabcbc".
> s = "3[a2[c]]", 返回 "accaccacc".
> s = "2[abc]3[cd]ef", 返回 "abcabccdcdcdef".

- 思路：通过栈来保存 `'['` 之前的数据

```Python
class Solution:
    def decodeString(self, s: str) -> str:
        stack, res, mul = [], "", 0
        
        for c in s:
            if c == "[":
                stack.append([res, mul])
                res, mul = "", 0
            elif c == "]":
                last_res, cur_mul = stack.pop()
                res = last_res + cur_mul * res
            elif c <= '9' and c >= '0':
                mul = mul * 10 + int(c)
            else:
                res += c
        
        return res
```

### [binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

> 给定一个二叉树，返回它的*中序*遍历。

方法1: recursion

```Python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        
        def inorder(cur):
            if not cur: return 
            
            inorder(cur.left)
            res.append(cur.val)
            inorder(cur.right)
        
        inorder(root)
        
        return res
```

方法2: 利用1个stack来实现inorder. [Reference](https://en.wikipedia.org/wiki/Tree_traversal#In-order) from wikipedia.

```Python
class Solution:
    def inorderTraversal(self, root: TreeNode) -> List[int]:
        
        stack, res = [], []
        node = root
 
        while len(stack) > 0 or node is not None:
            if node is not None: 
                stack.append(node)
                node = node.left
            else:
                node = stack.pop()
                res.append(node.val)
                node = node.right
        
        return res
```

### [clone-graph](https://leetcode-cn.com/problems/clone-graph/)

> 给你无向连通图中一个节点的引用，请你返回该图的深拷贝（克隆）。

- BFS

```Python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        lookup = {}
        def bfs(node):
            if not node: return node
            
            clone_node = Node(node.val, [])
            lookup[node] = clone_node
            queue = deque()
            queue.append(node)
            
            while queue:
                cur = queue.popleft()
                for n in cur.neighbors:
                    if n not in lookup:
                        lookup[n] = Node(n.val, [])
                        queue.append(n)
                    lookup[cur].neighbors.append(lookup[n])
            return clone_node
        
        return bfs(node)
```

- DFS

```Python
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        lookup = {}
        
        def dfs(node):
            if not node: return 
            
            if node in lookup:
                return lookup[node]
            
            clone_node = Node(node.val, [])
            lookup[node] = clone_node
            
            for n in node.neighbors:
                clone_node.neighbors.append(dfs(n))
            
            return clone_node
        
        return dfs(node)
```

### [number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)

> 给定一个由  '1'（陆地）和 '0'（水）组成的的二维网格，计算岛屿的数量。一个岛被水包围，并且它是通过水平方向或垂直方向上相邻的陆地连接而成的。你可以假设网格的四个边均被水包围。

High-level problem: number of connected component of graph

- 思路：通过深度搜索遍历可能性（注意标记已访问元素）

```Python
    def numIslands(self, grid: List[List[str]]) -> int:
        def dfs(grid, i, j):
            if i < 0 or i >= len(grid) or j < 0 or j >= len(grid[0]) or grid[i][j] == "0": return
            
            grid[i][j] = "0"
            dfs(grid, i - 1, j)
            dfs(grid, i + 1, j)
            dfs(grid, i, j -1)
            dfs(grid, i, j + 1)
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    dfs(grid, i, j)
                    count += 1
        
        return count
```

方法2:  BFS
```Python
class Solution:
    def numIslands(self, grid: [[str]]) -> int:
        def bfs(grid, i, j):
            queue = [[i, j]]
            while queue:
                [i, j] = queue.pop(0)
                if 0 <= i < len(grid) and 0 <= j < len(grid[0]) and grid[i][j] == '1':
                    grid[i][j] = '0'
                    queue += [[i + 1, j], [i - 1, j], [i, j - 1], [i, j + 1]]
        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == '0': continue
                bfs(grid, i, j)
                count += 1
        return count

```

### [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

> 给定 _n_ 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。
> 求在该柱状图中，能够勾勒出来的矩形的最大面积。

- 思路 1：蛮力法，比较每个以 i 开始 j 结束的最大矩形，A(i, j) = (j - i + 1) * min_height(i, j)，时间复杂度 O(n^2) 无法 AC。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        max_area = 0
        
        n = len(heights)
        for i in range(n):
            min_height = heights[i]
            for j in range(i, n):
                min_height = min(min_height, heights[j])
                max_area = max(max_area, min_height * (j - i + 1))
        
        return max_area
```

- 思路 2: 设 A(i, j) 为区间 [i, j) 内最大矩形的面积，k 为 [i, j) 内最矮 bar 的坐标，则 A(i, j) = max((j - i) * heights[k], A(i, k), A(k+1, j)), 使用分治法进行求解。时间复杂度 O(nlogn)，其中使用简单遍历求最小值无法 AC (最坏情况退化到 O(n^2))，使用线段树优化后勉强 AC。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        n = len(heights)
        
        seg_tree = [None] * n
        seg_tree.extend(list(zip(heights, range(n))))
        for i in range(n - 1, 0, -1):
            seg_tree[i] = min(seg_tree[2 * i], seg_tree[2 * i + 1], key=lambda x: x[0])
        
        def _min(i, j):
            min_ = (heights[i], i)
            i += n
            j += n
            while i < j:
                if i % 2 == 1:
                    min_ = min(min_, seg_tree[i], key=lambda x: x[0])
                    i += 1
                if j % 2 == 1:
                    j -= 1
                    min_ = min(min_, seg_tree[j], key=lambda x: x[0])
                i //= 2
                j //= 2
            
            return min_
        
        def LRA(i, j):
            if i == j:
                return 0
            min_k, k = _min(i, j)
            return max(min_k * (j - i), LRA(k + 1, j), LRA(i, k))
        
        return LRA(0, n)
```

- 思路 3：包含当前 bar 最大矩形的边界为左边第一个高度小于当前高度的 bar 和右边第一个高度小于当前高度的 bar。

```Python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        
        n = len(heights)
        
        stack = [-1]
        max_area = 0
        
        for i in range(n):
            while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                h = stack.pop()
                max_area = max(max_area, heights[h] * (i - stack[-1] - 1))
            stack.append(i)
        
        while len(stack) > 1:
            h = stack.pop()
            max_area = max(max_area, heights[h] * (n - stack[-1] - 1))
        
        return max_area
```

## Queue 队列

常用于 BFS 宽度优先搜索

### [implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)

> 使用栈实现队列

```Python
class MyQueue:

    def __init__(self):
        self.cache = []
        self.out = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.cache.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if len(self.out) == 0:
            while len(self.cache) > 0:
                self.out.append(self.cache.pop())

        return self.out.pop() 

    def peek(self) -> int:
        """
        Get the front element.
        """
        if len(self.out) > 0:
            return self.out[-1]
        else:
            return self.cache[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return len(self.cache) == 0 and len(self.out) == 0
```

### [binary-tree-level-order-traversal](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> 二叉树的层序遍历

```Python
class Solution:
    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        
        levels = []
        if root is None:
            return levels
        
        bfs = collections.deque([root])
        
        while len(bfs) > 0:
            levels.append([])
            
            level_size = len(bfs)
            for _ in range(level_size):
                node = bfs.popleft()
                levels[-1].append(node.val)
                
                if node.left is not None:
                    bfs.append(node.left)
                if node.right is not None:
                    bfs.append(node.right)
        
        return levels
```

### [01-matrix](https://leetcode-cn.com/problems/01-matrix/)

> 给定一个由 0 和 1 组成的矩阵，找出每个元素到最近的 0 的距离。
> 两个相邻元素间的距离为 1

- 思路 1: 从 0 开始 BFS, 遇到距离最小值需要更新的则更新后重新入队更新后续结点

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        M, N = len(matrix), len(matrix[0])
        queue = collections.deque()
        visited = [[0] * N for _ in range(M)]
        res = [[0] * N for _ in range(M)]
        for i in range(M):
            for j in range(N):
                if matrix[i][j] == 0:
                    queue.append((i, j))
                    visited[i][j] = 1
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        step = 0
        while queue:
            size = len(queue)
            for i in range(size):
                x, y = queue.popleft()
                if matrix[x][y] == 1:
                    res[x][y] = step
                for dx, dy in dirs:
                    newx, newy = x + dx, y + dy
                    if newx < 0 or newx >= M or newy < 0 or newy >= N or visited[newx][newy] == 1:
                        continue
                    queue.append((newx, newy))
                    visited[newx][newy] = 1
            step += 1
        return res  
```

- 思路 2: 2-pass DP，dist(i, j) = max{dist(i - 1, j), dist(i + 1, j), dist(i, j - 1), dist(i, j + 1)} + 1

```Python
class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        
        if len(matrix) == 0 or len(matrix[0]) == 0:
            return matrix
        
        m, n = len(matrix), len(matrix[0])
        
        dist = [[float('inf')] * n for _ in range(m)]
        
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 1:
                    if i - 1 >= 0:
                        dist[i][j] = min(dist[i - 1][j] + 1, dist[i][j])
                    if j - 1 >= 0:
                        dist[i][j] = min(dist[i][j - 1] + 1, dist[i][j])
                else:
                    dist[i][j] = 0
        
        for i in range(-1, -m - 1, -1):
            for j in range(-1, -n - 1, -1):
                if matrix[i][j] == 1:
                    if i + 1 < 0:
                        dist[i][j] = min(dist[i + 1][j] + 1, dist[i][j])
                    if j + 1 < 0:
                        dist[i][j] = min(dist[i][j + 1] + 1, dist[i][j])
        
        return dist
```

## 补充：单调栈

顾名思义，单调栈即是栈中元素有单调性的栈，典型应用为用线性的时间复杂度找左右两侧第一个大于/小于当前元素的位置。

### [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

```Python
class Solution:
    def largestRectangleArea(self, heights) -> int:
        heights.append(0)
        stack = [-1]
        result = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                cur = stack.pop()
                result = max(result, heights[cur] * (i - stack[-1] - 1))
            stack.append(i)
        return result
```

### [trapping-rain-water](https://leetcode-cn.com/problems/trapping-rain-water/)

```Python
class Solution:
    def trap(self, height: List[int]) -> int:
        
        stack = []
        result = 0
        
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                cur = stack.pop()
                if not stack:
                    break
                result += (min(height[stack[-1]], height[i]) - height[cur]) * (i - stack[-1] - 1)
            stack.append(i)
        
        return result
```

## 补充：单调队列

单调栈的拓展，可以从数组头 pop 出旧元素，典型应用是以线性时间获得区间最大/最小值。

### [sliding-window-maximum](https://leetcode-cn.com/problems/sliding-window-maximum/)

> 求滑动窗口中的最大元素

```Python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        
        N = len(nums)
        if N * k == 0:
            return []
        
        if k == 1:
            return nums[:]
        
        # define a max queue
        maxQ = collections.deque()
        
        result = []
        for i in range(N):
            if maxQ and maxQ[0] == i - k:
                maxQ.popleft()
            
            while maxQ and nums[maxQ[-1]] < nums[i]:
                maxQ.pop()
            
            maxQ.append(i)
            
            if i >= k - 1:
                result.append(nums[maxQ[0]])
        
        return result
```

### [shortest-subarray-with-sum-at-least-k](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/)

```Python
class Solution:
    def shortestSubarray(self, A: List[int], K: int) -> int:
        N = len(A)
        cdf = [0]
        for num in A:
            cdf.append(cdf[-1] + num)

        result = N + 1
        minQ = collections.deque()
        
        for i, csum in enumerate(cdf):
            
            while minQ and csum <= cdf[minQ[-1]]:
                minQ.pop()

            while minQ and csum - cdf[minQ[0]] >= K:
                result = min(result, i - minQ.popleft())

            minQ.append(i)

        return result if result < N + 1 else -1 
```

## 总结

- 熟悉栈的使用场景
  - 后入先出，保存临时值
  - 利用栈 DFS 深度搜索
- 熟悉队列的使用场景
  - 利用队列 BFS 广度搜索

## 练习

- [x] [min-stack](https://leetcode-cn.com/problems/min-stack/)
- [x] [evaluate-reverse-polish-notation](https://leetcode-cn.com/problems/evaluate-reverse-polish-notation/)
- [x] [decode-string](https://leetcode-cn.com/problems/decode-string/)
- [x] [binary-tree-inorder-traversal](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)
- [x] [clone-graph](https://leetcode-cn.com/problems/clone-graph/)
- [x] [number-of-islands](https://leetcode-cn.com/problems/number-of-islands/)
- [ ] [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
- [x] [implement-queue-using-stacks](https://leetcode-cn.com/problems/implement-queue-using-stacks/)
- [x] [01-matrix](https://leetcode-cn.com/problems/01-matrix/)
- [ ] [largest-rectangle-in-histogram](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)
- [ ] [trapping-rain-water](https://leetcode-cn.com/problems/trapping-rain-water/)
- [ ] [sliding-window-maximum](https://leetcode-cn.com/problems/sliding-window-maximum/)
- [ ] [shortest-subarray-with-sum-at-least-k](https://leetcode-cn.com/problems/shortest-subarray-with-sum-at-least-k/)
