# 回溯法

## 背景

回溯法（backtrack）常用于遍历列表所有子集，是 DFS 深度搜索一种，一般用于全排列，穷尽所有可能，遍历的过程实际上是一个决策树的遍历过程。时间复杂度一般 O(N!)，它不像动态规划存在重叠子问题可以优化，回溯算法就是纯暴力穷举，复杂度一般都很高。

## 模板

```python
res = []
path = []

def backtrack(未探索区域, res, path):
    if 未探索区域满足结束条件:
        res.add(path) # 深度拷贝
        return
    for 选择 in 未探索区域当前可能的选择:
        if 当前选择符合要求:
            path.add(当前选择)
            backtrack(新的未探索区域, res, path)
            path.pop()

```

核心就是从选择列表里做一个选择，然后一直递归往下搜索答案，如果遇到路径不通，就返回来撤销这次选择。

## 示例

### [subsets](https://leetcode-cn.com/problems/subsets/)

> 给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

遍历过程

```Python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)

        def helper(i, temp):
            res.append(temp)

            for j in range(i, n):
                helper(j + 1, temp + [nums[j]])
        

        helper(0, [])
        return res       
```

### [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)

> 给定一个可能包含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。说明：解集不能包含重复的子集。

```Python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        
        nums = sorted(nums)
        n = len(nums)
        result = []
        
        def backtrack(start, k, route=[]):
            
            if len(route) == k:
                result.append(route.copy())
                return
            
            last = None
            for i in range(start, n):
                if nums[i] != last:
                    route.append(nums[i])
                    backtrack(i + 1, k)
                    last = route.pop()
            
            return
        
        for k in range(n + 1):
            backtrack(0, k)
        
        return result
```

### [permutations](https://leetcode-cn.com/problems/permutations/)

> 给定一个没有重复数字的序列，返回其所有可能的全排列。

- 思路 1：需要记录已经选择过的元素，满足条件的结果才进行返回，需要额外 O(n) 的空间

```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        n = len(nums)
        result = []
        
        in_route = [False] * n
        
        def backtrack(route=[]):
            
            if len(route) == n:
                result.append(route.copy())
                return
                
            for i in range(n):
                if not in_route[i]:
                    route.append(nums[i])
                    in_route[i] = True
                    backtrack()
                    route.pop()
                    in_route[i] = False
            
            return
        
        backtrack()
        return result
```

- 思路 2: 针对此题的更高级的回溯，利用原有的数组，每次回溯将新选择的元素与当前位置元素交换，回溯完成再换回来

```Python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        n = len(nums)
        result = []
        
        def backtrack(idx=0):
            if idx == n:
                result.append(nums.copy())
            for i in range(idx, n):
                nums[idx], nums[i] = nums[i], nums[idx]
                backtrack(idx + 1)
                nums[idx], nums[i] = nums[i], nums[idx]
            return

        backtrack()
        return result
```

### [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)

> 给定一个可包含重复数字的序列，返回所有不重复的全排列。

注意此题（貌似）无法使用上题的思路 2，因为交换操作会打乱排序。

```Python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        n = len(nums)
        res = []
        in_route = [False] * n

        def backtrack(route=[]):
            if len(route) == n and route not in res:
                res.append(route.copy())
                return
            
            for i in range(n):
                if not in_route[i]:
                    route.append(nums[i])
                    in_route[i] = True
                    backtrack()
                    route.pop()
                    in_route[i] = False
            
            return
        
        backtrack()
        return res
```

### [combination-sum](https://leetcode-cn.com/problems/combination-sum/)

```Python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        
        n = len(candidates)
        result = []
        
        def backtrack(first=0, route=[], route_sum=0):
            
            if route_sum == target:
                result.append(route.copy())
                return
            
            if route_sum > target:
                return
            
            for i in range(first, n):
                route.append(candidates[i])
                route_sum += candidates[i]
                backtrack(i, route, route_sum)
                route_sum -= route.pop()
            
            return
        
        backtrack()
        return result
```

### [letter-combinations-of-a-phone-number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```Python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        
        n = len(digits)
        result = []
        
        if n == 0:
            return result
        
        num2char = {
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        
        def backtrack(idx=0, route=[]):
            if idx == n:
                result.append(''.join(route))
                return
            
            for c in num2char[digits[idx]]:
                route.append(c)
                backtrack(idx + 1, route)
                route.pop()
            
            return
        
        backtrack()
        return result
```

### [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)

方法1: 由于path + `[s[:i]]` 会产生新的list, 所以不需要copy
```Python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def isPalindrome(s):
            return s == s[::-1]

        def backtrack(s, path):
            if not s:
                res.append(path)
                return 

            for i in range(1, len(s) + 1):
                if isPalindrome(s[:i]):
                    backtrack(s[i:], path + [s[:i]])

        backtrack(s, [])

        return res
```

```Python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        
        N = len(s)
        Pal = collections.defaultdict(set)
        
        def isPal(i, j):
            if i < j:
                return j in Pal[i]
            return True
        
        for j in range(N):
            for i in range(j + 1):
                if s[i] == s[j] and isPal(i + 1, j - 1):
                    Pal[i].add(j)
        
        result = []
        
        def backtrack(first=0, route=[]):

            if first == N:
                result.append(route[:])
                return

            for i in Pal[first]:
                route.append(s[first:i+1])
                backtrack(i + 1)
                route.pop()
            
            return
        
        backtrack()
        return result
```

### [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)

```Python
class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        
        n = len(s)
        result = []
        
        if n > 12:
            return result
        
        def Valid_s(i, j):
            return i < j and j <= n and ((s[i] != '0' and int(s[i:j]) < 256) or (s[i] == '0' and i == j - 1))
        
        def backtrack(start=0, route=[]):
            
            if len(route) == 3:
                if Valid_s(start, n):
                    result.append('.'.join(route) + '.' + s[start:])
                return
            
            for i in range(start, start + 3):
                if Valid_s(start, i + 1):
                    route.append(s[start:i + 1])
                    backtrack(i + 1, route)
                    route.pop()
                
            return
        
        backtrack()
        return result
```



## 练习

- [x] [subsets](https://leetcode-cn.com/problems/subsets/)
- [x] [subsets-ii](https://leetcode-cn.com/problems/subsets-ii/)
- [x] [permutations](https://leetcode-cn.com/problems/permutations/)
- [x] [permutations-ii](https://leetcode-cn.com/problems/permutations-ii/)

- [x] [combination-sum](https://leetcode-cn.com/problems/combination-sum/)
- [x] [letter-combinations-of-a-phone-number](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)
- [x] [palindrome-partitioning](https://leetcode-cn.com/problems/palindrome-partitioning/)
- [x] [restore-ip-addresses](https://leetcode-cn.com/problems/restore-ip-addresses/)
