//
//  SlideWindow.swift
//  DataStructDemo
//
//  Created by 刘练 on 2020/9/6.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

public class TreeNode {
    public var val: Int
    public var left: TreeNode?
    public var right: TreeNode?
    public init(_ val: Int) {
         self.val = val
         self.left = nil
         self.right = nil
    }
}

class Homework_Week_02 {
    /**
     242. 有效的字母异位词
     
     给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

     示例 1:

     输入: s = "anagram", t = "nagaram"
     输出: true
     示例 2:

     输入: s = "rat", t = "car"
     输出: false
     说明:
     你可以假设字符串只包含小写字母。

     进阶:
     如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-anagram
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isAnagram(_ s: String, _ t: String) -> Bool {
        if s.count != t.count {
            return false
        }
        
        if s.count > 0 && t.count > 0 {
            let sArr = Array(s)
            var dict = [Character : Int]()
            for char in sArr {
                if let val = dict[char] {
                    dict[char] = val + 1
                } else {
                    dict[char] = 1
                }
            }
            
            let tArr = Array(t)
            for char in tArr {
                if let val = dict[char] {
                    dict[char] = val - 1
                }
            }
            
            for val in dict.values {
                if 0 != val {
                    return false
                }
            }
            
            return true
        }
        
        return true
    }
    
    /**
     49. 字母异位词分组
     
     给定一个字符串数组，将字母异位词组合在一起。字母异位词指字母相同，但排列不同的字符串。

     示例:

     输入: ["eat", "tea", "tan", "ate", "nat", "bat"]
     输出:
     [
       ["ate","eat","tea"],
       ["nat","tan"],
       ["bat"]
     ]
     说明：

     所有输入均为小写字母。
     不考虑答案输出的顺序。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/group-anagrams
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func groupAnagrams(_ strs: [String]) -> [[String]] {
        var sortedStrs = [String]()
        for str in strs {
            sortedStrs.append(String(str.sorted()))
        }
        
        var dict = [String : [Int]]()
        for i in 0 ..< sortedStrs.count {
            let str = sortedStrs[i]
            if var arr = dict[str] {
                arr.append(i)
                dict[str] = arr
            } else {
                var indexs = [Int]()
                indexs.append(i)
                dict[str] = indexs
            }
        }
        
        var res = [[String]]()
        for key in dict.keys {
            var arr = [String]()
            for index in dict[key]! {
                arr.append(strs[index])
            }
            res.append(arr)
        }
        
        return res
    }
    
    /**
     给定一个整数数组 nums 和一个目标值 target，请你在该数组中找出和为目标值的那 两个 整数，并返回他们的数组下标。

     你可以假设每种输入只会对应一个答案。但是，数组中同一个元素不能使用两遍。


     示例:

     给定 nums = [2, 7, 11, 15], target = 9

     因为 nums[0] + nums[1] = 2 + 7 = 9
     所以返回 [0, 1]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/two-sum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func twoSum(_ nums: [Int], _ target: Int) -> [Int] {
        var result = [Int]()
        if nums.count > 1 {
            var dict = [Int : Int]()
            for i in 0 ..< nums.count {
                dict[nums[i]] = i
            }
            
            for i in 0 ..< nums.count {
                let val = target - nums[i]
                if dict.keys.contains(val) && i != dict[val]! {
                    if dict[val]! > i {
                        result.append(i)
                        result.append(dict[val]!)
                    } else {
                        result.append(dict[val]!)
                        result.append(i)
                    }
                    break
                }
            }
        }
        return result
    }
    
    /**
     94. 二叉树的中序遍历
     
     给定一个二叉树，返回它的中序 遍历。

     示例:

     输入: [1,null,2,3]
        1
         \
          2
         /
        3

     输出: [1,3,2]
     进阶: 递归算法很简单，你可以通过迭代算法完成吗？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/binary-tree-inorder-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func inorderTraversal(_ root: TreeNode?) -> [Int] {
        var res = [Int]()
        var stack = [TreeNode]()
        var node = root
        while !stack.isEmpty || nil != node {
            while nil != node {
                stack.append(node!)
                node = node!.left
            }

            node = stack.removeLast()
            res.append(node!.val)
            node = node!.right
        }
        return res
    }
    
    /**
     144. 二叉树的前序遍历
     
     给定一个二叉树，返回它的 前序 遍历。

      示例:

     输入: [1,null,2,3]
        1
         \
          2
         /
        3

     输出: [1,2,3]
     进阶: 递归算法很简单，你可以通过迭代算法完成吗？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/binary-tree-preorder-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func preorderTraversal(_ root: TreeNode?) -> [Int] {
        var res = [Int]()
        var stack = [TreeNode]()
        if nil != root {
            stack.append(root!)
        }

        while !stack.isEmpty {
            let node = stack.removeLast()
            res.append(node.val)

            if let right = node.right {
                stack.append(right)
            }

            if let left = node.left {
                stack.append(left)
            }
        }

        return res
    }
    
    /**
     589. N叉树的前序遍历
     
     给定一个 N 叉树，返回其节点值的前序遍历。

     例如，给定一个 3叉树 :

     返回其前序遍历: [1,3,5,6,2,4]。

      

     说明: 递归法很简单，你可以使用迭代法完成此题吗?

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/n-ary-tree-preorder-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func preorder(_ root: Node?) -> [Int] {
        var res = [Int]()
        var stack = [Node]()
        if nil != root {
            stack.append(root!)
        }
        while !stack.isEmpty {
            let node = stack.removeLast()
            res.append(node.val)
            for i in (0 ..< node.children.count).reversed() {
                stack.append(node.children[i])
            }
        }
        return res
    }
    
    /**
     429. N叉树的层序遍历
     
     给定一个 N 叉树，返回其节点值的层序遍历。 (即从左到右，逐层遍历)。

     例如，给定一个 3叉树 :
      

     返回其层序遍历:

     [
          [1],
          [3,2,4],
          [5,6]
     ]
      

     说明:

     树的深度不会超过 1000。
     树的节点总数不会超过 5000。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func levelOrder(_ root: Node?) -> [[Int]] {
        var res = [[Int]]()
        var queue = [Node]()
        if nil != root {
            queue.append(root!)
        }
        while !queue.isEmpty {
            let size = queue.count
            var levelRes = [Int]()
            for _ in 0 ..< size {
                let node = queue.removeFirst()
                levelRes.append(node.val)
                for child in node.children {
                    queue.append(child)
                }
            }
            res.append(levelRes)
        }
        return res
    }
    
    /**
     590. N叉树的后序遍历
     
     给定一个 N 叉树，返回其节点值的后序遍历。

     例如，给定一个 3叉树 :

     返回其后序遍历: [5,6,3,2,4,1].

      

     说明: 递归法很简单，你可以使用迭代法完成此题吗?

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/n-ary-tree-postorder-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func postorder(_ root: Node?) -> [Int] {
        // 递归解法
        /**
         var res = [Int]()
         postorderHelper(root, &res)
         return res
         */
        
        // 迭代解法： 根 - 右 - 左 遍历之后翻转即为后序遍历即 左 - 右 - 根
        var res = [Int]()
        var stack = [Node]()
        if nil != root {
            stack.append(root!)
        }
        
        while !stack.isEmpty {
            let node = stack.removeLast()
            res.insert(node.val, at: 0)
            for child in node.children {
                stack.append(child)
            }
        }
        
        return res
    }
    
    private func postorderHelper(_ root: Node?, _ res: inout [Int]) {
        if let children = root?.children {
            for node in children {
                postorderHelper(node, &res)
            }
        }
        
        if let node = root {
            res.append(node.val)
        }
    }
    
    /**
    40. 最小的k个数
    
    输入整数数组 arr ，找出其中最小的 k 个数。例如，输入4、5、1、6、2、7、3、8这8个数字，则最小的4个数字是1、2、3、4。

     

    示例 1：

    输入：arr = [3,2,1], k = 2
    输出：[1,2] 或者 [2,1]
    示例 2：

    输入：arr = [0,1,2,1], k = 1
    输出：[0]
     

    限制：

    0 <= k <= arr.length <= 10000
    0 <= arr[i] <= 10000

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/zui-xiao-de-kge-shu-lcof
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    */
    func getLeastNumbers(_ arr: [Int], _ k: Int) -> [Int] {
        var heap = Heap<Int>(elements: []) { (a, b) -> Bool in
            return a < b
        }
        for val in arr {
            heap.add(val)
        }

        var res = [Int]()
        for _ in 0 ..< k {
            res.append(heap.poll()!)
        }
        return res
    }
    
    /**
     239. 滑动窗口最大值
     
     给定一个数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。滑动窗口每次只向右移动一位。

     返回滑动窗口中的最大值。

      

     进阶：

     你能在线性时间复杂度内解决此题吗？

      

     示例:

     输入: nums = [1,3,-1,-3,5,3,6,7], 和 k = 3
     输出: [3,3,5,5,6,7]
     解释:

       滑动窗口的位置                最大值
     ---------------               -----
     [1  3  -1] -3  5  3  6  7       3
      1 [3  -1  -3] 5  3  6  7       3
      1  3 [-1  -3  5] 3  6  7       5
      1  3  -1 [-3  5  3] 6  7       5
      1  3  -1  -3 [5  3  6] 7       6
      1  3  -1  -3  5 [3  6  7]      7
      

     提示：

     1 <= nums.length <= 10^5
     -10^4 <= nums[i] <= 10^4
     1 <= k <= nums.length

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sliding-window-maximum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        if k < 2 {
            return nums
        }
        
        if k > nums.count {
            var max = Int.min
            for num in nums {
                if num > max {
                    max = num
                }
            }
            return [max]
        }
        
        var result = [Int]()
        var deque = [Int]()
        for i in 0 ..< k {
            while !deque.isEmpty && nums[i] > nums[deque.last!] {
                deque.removeLast()
            }
            deque.append(i)
        }
        result.append(nums[deque.first!])
        
        for i in k ..< nums.count {
            // Remove element not in window
            while !deque.isEmpty && deque.first! <= i - k {
                deque.removeFirst()
            }
            
            while !deque.isEmpty && nums[i] > nums[deque.last!] {
                deque.removeLast()
            }
            
            deque.append(i)
            result.append(nums[deque.first!])
        }
        
        return result
    }
    
    /**
     剑指 Offer 49. 丑数
     
     我们把只包含质因子 2、3 和 5 的数称作丑数（Ugly Number）。求按从小到大的顺序的第 n 个丑数。

      

     示例:

     输入: n = 10
     输出: 12
     解释: 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 是前 10 个丑数。
     说明:

     1 是丑数。
     n 不超过1690。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/chou-shu-lcof
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func nthUglyNumber(_ n: Int) -> Int {
        if n > 0 {
            var dp = [Int](repeating: 1, count: n)
            var a = 0, b = 0, c = 0
            for i in 1 ..< n {
                let n2 = dp[a] * 2, n3 = dp[b] * 3, n5 = dp[c] * 5
                dp[i] = min(n2, n3, n5)
                if n2 == dp[i] {
                    a += 1
                }
                if n3 == dp[i] {
                    b += 1
                }
                if n5 == dp[i] {
                    c += 1
                }
            }
            return dp[n - 1]
        }
        
        return 1
    }
    
    /**
     347. 前 K 个高频元素
     
     给定一个非空的整数数组，返回其中出现频率前 k 高的元素。

      

     示例 1:

     输入: nums = [1,1,1,2,2,3], k = 2
     输出: [1,2]
     示例 2:

     输入: nums = [1], k = 1
     输出: [1]
      

     提示：

     你可以假设给定的 k 总是合理的，且 1 ≤ k ≤ 数组中不相同的元素的个数。
     你的算法的时间复杂度必须优于 O(n log n) , n 是数组的大小。
     题目数据保证答案唯一，换句话说，数组中前 k 个高频元素的集合是唯一的。
     你可以按任意顺序返回答案。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/top-k-frequent-elements
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
        var dict = [Int : Int]()
        for val in nums {
            if let count = dict[val] {
                dict[val] = count + 1
            } else {
                dict[val] = 1
            }
        }
                
        var heap = Heap<(Int, Int)>(elements: []) { (a, b) -> Bool in
            return a.0 > b.0
        }
        
        for key in dict.keys {
            heap.add((dict[key]!, key))
        }
        
        var res = [Int]()
        for _ in 0 ..< k {
            if heap.count > 0 {
                res.append(heap.poll()!.1)
            } else {
                break
            }
        }
        return res
    }
    
    /**
     假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

     每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

     注意：给定 n 是一个正整数。

     示例 1：

     输入： 2
     输出： 2
     解释： 有两种方法可以爬到楼顶。
     1.  1 阶 + 1 阶
     2.  2 阶
     示例 2：

     输入： 3
     输出： 3
     解释： 有三种方法可以爬到楼顶。
     1.  1 阶 + 1 阶 + 1 阶
     2.  1 阶 + 2 阶
     3.  2 阶 + 1 阶

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/climbing-stairs
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     
     f(n) = f(n - 1) + f(n - 2)
     */
    func climbStairs(_ n: Int) -> Int {
        if n <= 0 {
            return 0
        }
        
        if 1 == n {
            return 1
        }
        
        if 2 == n {
            return 2
        }
        
        var pre = 1
        var next = 2
        var sum = 0
        for _ in 3 ... n {
            sum = pre + next
            pre = next
            next = sum
        }
        
        return sum
    }
    
    /**
     22. 括号生成
     
     数字 n 代表生成括号的对数，请你设计一个函数，用于能够生成所有可能的并且 有效的 括号组合。

      

     示例：

     输入：n = 3
     输出：[
            "((()))",
            "(()())",
            "(())()",
            "()(())",
            "()()()"
          ]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/generate-parentheses
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private var parenthesis: [String] = [String]()
    func generateParenthesis(_ n: Int) -> [String] {
        self.generateParenthesisRecursive(0, 0, n, "")
        return parenthesis
    }
    
    private func generateParenthesisRecursive(_ left: Int, _ right: Int, _ max: Int, _ string: String) {
        // terminator
        if left == max && right == max {
            self.parenthesis.append(string)
            return
        }
        
        // process current logic
        let leftStr = string + "("
        let rightStr = string + ")"
        
        // drill down
        if left < max {
            generateParenthesisRecursive(left + 1, right, max, leftStr)
        }
        
        if left > right {
            generateParenthesisRecursive(left, right + 1, max, rightStr)
        }
        
        // restore state
    }
    
    /**
     50. Pow(x, n)
     
     实现 pow(x, n) ，即计算 x 的 n 次幂函数。

     示例 1:

     输入: 2.00000, 10
     输出: 1024.00000
     示例 2:

     输入: 2.10000, 3
     输出: 9.26100
     示例 3:

     输入: 2.00000, -2
     输出: 0.25000
     解释: 2-2 = 1/22 = 1/4 = 0.25
     说明:

     -100.0 < x < 100.0
     n 是 32 位有符号整数，其数值范围是 [−231, 231 − 1] 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/powx-n
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func myPow(_ x: Double, _ n: Int) -> Double {
        if 0 == n {
            return 1.0
        }
        
        if n > 0 {
            return myPowHelper(x, n)
        } else {
            return 1.0/myPowHelper(x, -n)
        }
    }
    
    private func myPowHelper(_ x: Double, _ n: Int) -> Double {
        // terminator
        if 0 == n {
            return 1.0
        }
        
        // process current logic and drill down
        var res = myPowHelper(x, n >> 1)
        if 1 == n % 2 {
            res = res * res * x
        } else {
            res = res * res
        }
        
        return res
    }
    
    /**
     78. 子集
     
     给定一组不含重复元素的整数数组 nums，返回该数组所有可能的子集（幂集）。

     说明：解集不能包含重复的子集。

     示例:

     输入: nums = [1,2,3]
     输出:
     [
       [3],
       [1],
       [2],
       [1,2,3],
       [1,3],
       [2,3],
       [1,2],
       []
     ]


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/subsets
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    var subsetsRes = [[Int]]()
    var k = 0
    func subsets(_ nums: [Int]) -> [[Int]] {
        var res = [Int]()
        while k <= nums.count {
            backtrack(0, &res, nums)
            k += 1
        }
        return subsetsRes
    }
    
    /**
     解决一个回溯问题，实际上就是一个决策树的遍历过程。只需要思考 3 个问题：
     1、路径：也就是已经做出的选择。
     2、选择列表：也就是你当前可以做的选择。
     3、结束条件：也就是到达决策树底层，无法再做选择的条件。
     
     代码框架：
     
     result = []
     def backtrack(路径, 选择列表):
         if 满足结束条件:
             result.add(路径)
             return

         for 选择 in 选择列表:
             做选择
             backtrack(路径, 选择列表)
             撤销选择
     
     其核心就是 for 循环里面的递归，在递归调用之前「做选择」，在递归调用之后「撤销选择」
     */
    private func backtrack(_ first: Int, _ curr: inout [Int], _ nums: [Int]) {
        if k == curr.count {
            subsetsRes.append(curr)
        }
        
        for i in first ..< nums.count {
            curr.append(nums[i])
            backtrack(i + 1, &curr, nums)
            curr.remove(at: curr.count - 1)
        }
    }
    
    /**
     51. N 皇后
     
     n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。



     上图为 8 皇后问题的一种解法。

     给定一个整数 n，返回所有不同的 n 皇后问题的解决方案。

     每一种解法包含一个明确的 n 皇后问题的棋子放置方案，该方案中 'Q' 和 '.' 分别代表了皇后和空位。

      

     示例：

     输入：4
     输出：[
      [".Q..",  // 解法 1
       "...Q",
       "Q...",
       "..Q."],

      ["..Q.",  // 解法 2
       "Q...",
       "...Q",
       ".Q.."]
     ]
     解释: 4 皇后问题存在两个不同的解法。
      

     提示：

     皇后彼此不能相互攻击，也就是说：任何两个皇后都不能处于同一条横行、纵行或斜线上。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/n-queens
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    var queens = [[String]]()
    func solveNQueens(_ n: Int) -> [[String]] {
        var board = [[String]]()
        for _ in 0 ..< n {
            let string = [String](repeating: ".", count: n)
            board.append(string)
        }
        nqueensBacktrack(&board, 0)
        return queens
    }
    
    /**
     路径：board 中小于 row 的那些行都已经成功放置了皇后
     选择列表：第 row 行的所有列都是放置皇后的选择
     结束条件：row 超过 board 的最后一行
     */
    private func nqueensBacktrack(_ board: inout [[String]], _ row: Int) {
        // 结束条件
        if row == board.count {
            queens.append(convertBoard(board))
            return
        }
        
        let n = board[row].count
        for col in 0 ..< n {
            // 排除不合法选择
            if !isValid(board, row, col) {
                continue
            }
            
            // 做选择
            board[row][col] = "Q"
            // 进入下一行决策
            nqueensBacktrack(&board, row + 1)
            // 撤销选择
            board[row][col] = "."
        }
    }
    
    private func convertBoard(_ board: [[String]]) -> [String] {
        var res = [String]()
        for i in 0 ..< board.count {
            var str = ""
            for j in 0 ..< board[i].count {
                str += board[i][j]
            }
            res.append(str)
        }
        return res
    }
    
    /**
     是否可以在 board[row][col] 放置皇后
     */
    private func isValid(_ board: [[String]], _ row: Int, _ col: Int) -> Bool {
        let n = board.count
        // 检查列是否有皇后冲突
        for i in 0 ..< n {
            if board[i][col] == "Q" {
                return false
            }
        }
        
        // 检查右上方是否有皇后冲突
        var i = row - 1, j = col + 1
        while i >= 0 && j >= 0 && j < n {
            if board[i][j] == "Q" {
                return false
            }
            
            i -= 1
            j += 1
        }
        
        // 检查左上方是否有皇后冲突
        i = row - 1
        j = col - 1
        while i >= 0 && j >= 0 {
            if board[i][j] == "Q" {
                return false
            }
            
            i -= 1
            j -= 1
        }
        
        return true
    }
}

/**
 Swift 实现堆
 */

struct Heap<Element> {
    private var elements: [Element]
    private let priorityFunction: (Element, Element) -> Bool
    
    var count: Int {
        return self.elements.count
    }
    
    var isEmpty: Bool {
        return self.elements.isEmpty
    }
    
    init(elements: [Element] = [], priorityFunction: @escaping (Element, Element) -> Bool) {
        self.elements = elements
        self.priorityFunction = priorityFunction
    }
        
    private func parentIndex(of index: Int) -> Int {
        return (index - 1) >> 1
    }
    
    private func leftChildIndex(of index: Int) -> Int {
        return 2 * index + 1
    }
    
    private func rightChildIndex(of index: Int) -> Int {
        return 2 * index + 2
    }
    
    private func printHeap() {
        print("----- print heap ------\r\n")
        for element in self.elements {
            print("\(element) ")
        }
        print("----- print heap ------\r\n")
    }
    
    mutating func add(_ element: Element) {
        self.offer(element)
    }
    
    mutating func offer(_ element: Element) {
        self.elements.append(element)
        self.siftUp(self.count - 1)
    }
    
    func peek() -> Element? {
        return self.isEmpty ? nil : self.elements.first
    }
    
    mutating func poll() -> Element? {
        if self.isEmpty {
            return nil
        }
        
        let result = self.elements.first
        self.elements.swapAt(0, self.count - 1)
        self.elements.removeLast()
        self.siftDown(0)
        return result
    }
}

extension Heap {
    func isHigherPriority(firstElement: Element, secondElement: Element) -> Bool {
        return self.priorityFunction(firstElement, secondElement)
    }
    
    mutating func siftUp(_ index: Int) {
        var k = index
        let element = self.elements[k]
        while k > 0 {
            let parentIndex = self.parentIndex(of: k)
            let parentElement = self.elements[parentIndex]
            if self.isHigherPriority(firstElement: parentElement, secondElement: element) {
                break
            }
            self.elements.swapAt(k, parentIndex)
            k = parentIndex
        }
    }
    
    mutating func siftDown(_ index: Int) {
        guard self.count > index else {
            return
        }
        
        var k = index
        let element = self.elements[k]
        let half = self.count >> 1
        while k < half {
            var leftIndex = self.leftChildIndex(of: k)
            let rightIndex = self.rightChildIndex(of: k)
            if rightIndex < self.count && self.isHigherPriority(firstElement: self.elements[rightIndex], secondElement: self.elements[leftIndex]) {
                leftIndex = rightIndex
            }
            
            if self.isHigherPriority(firstElement: element, secondElement: self.elements[leftIndex]) {
                break
            }
            self.elements.swapAt(k, leftIndex)
            k = leftIndex
        }
    }
}
