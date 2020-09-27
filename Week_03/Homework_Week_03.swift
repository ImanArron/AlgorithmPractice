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

class Homework_Week_03 {
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
     226. 翻转二叉树
     
     翻转一棵二叉树。

     示例：

     输入：

          4
        /   \
       2     7
      / \   / \
     1   3 6   9
     输出：

          4
        /   \
       7     2
      / \   / \
     9   6 3   1
     备注:
     这个问题是受到 Max Howell 的 原问题 启发的 ：

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/invert-binary-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func invertTree(_ root: TreeNode?) -> TreeNode? {
        if nil == root {
            return nil
        }

        let left = root?.left
        let right = root?.right
        root?.right = left
        root?.left = right
        
        invertTree(root?.left)
        invertTree(root?.right)

        return root
    }
    
    /**
     98. 验证二叉搜索树
     
     给定一个二叉树，判断其是否是一个有效的二叉搜索树。

     假设一个二叉搜索树具有如下特征：

     节点的左子树只包含小于当前节点的数。
     节点的右子树只包含大于当前节点的数。
     所有左子树和右子树自身必须也是二叉搜索树。
     示例 1:

     输入:
         2
        / \
       1   3
     输出: true
     示例 2:

     输入:
         5
        / \
       1   4
          / \
         3   6
     输出: false
     解释: 输入为: [5,1,4,null,null,3,6]。
          根节点的值为 5 ，但是其右子节点值为 4 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/validate-binary-search-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private var val = Int.min
    func isValidBST(_ root: TreeNode?) -> Bool {
//         if nil == root {
//             return true
//         }
//
//         if !isValidBST(root?.left) {
//             return false
//         }
//
//         if root!.val <= val {
//             return false
//         }
//         val = root!.val
//
//         return isValidBST(root?.right)
        return isValidBSTHelper(root, Int.min, Int.max)
    }

    private func isValidBSTHelper(_ root: TreeNode?, _ lower: Int, _ upper: Int) -> Bool {
        if nil == root {
            return true
        }

        let val = root!.val
        if val <= lower || val >= upper {
            return false
        }

        return isValidBSTHelper(root!.left, lower, val) && isValidBSTHelper(root!.right, val, upper)
    }
    
    /**
     104. 二叉树的最大深度
     
     给定一个二叉树，找出其最大深度。

     二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

     说明: 叶子节点是指没有子节点的节点。

     示例：
     给定二叉树 [3,9,20,null,null,15,7]，

         3
        / \
       9  20
         /  \
        15   7
     返回它的最大深度 3 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximum-depth-of-binary-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxDepth(_ root: TreeNode?) -> Int {
        if nil == root {
            return 0
        }

        return max(maxDepth(root?.left), maxDepth(root?.right)) + 1
    }
    
    /**
     111. 二叉树的最小深度
     
     给定一个二叉树，找出其最小深度。

     最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

     说明: 叶子节点是指没有子节点的节点。

     示例:

     给定二叉树 [3,9,20,null,null,15,7],

         3
        / \
       9  20
         /  \
        15   7
     返回它的最小深度  2.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/minimum-depth-of-binary-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minDepth(_ root: TreeNode?) -> Int {
        if nil == root {
            return 0
        }

        let leftDepth = minDepth(root?.left)
        let rightDepth = minDepth(root?.right)
        if nil == root?.left || nil == root?.right {
            return leftDepth + rightDepth + 1
        } else {
            return min(leftDepth, rightDepth) + 1
        }
    }
    
    /**
     297. 二叉树的序列化与反序列化
     
     序列化是将一个数据结构或者对象转换为连续的比特位的操作，进而可以将转换后的数据存储在一个文件或者内存中，同时也可以通过网络传输到另一个计算机环境，采取相反方式重构得到原数据。

     请设计一个算法来实现二叉树的序列化与反序列化。这里不限定你的序列 / 反序列化算法执行逻辑，你只需要保证一个二叉树可以被序列化为一个字符串并且将这个字符串反序列化为原始的树结构。

     示例:

     你可以将以下二叉树：

         1
        / \
       2   3
          / \
         4   5

     序列化为 "[1,2,3,null,null,4,5]"
     提示: 这与 LeetCode 目前使用的方式一致，详情请参阅 LeetCode 序列化二叉树的格式。你并非必须采取这种方式，你也可以采用其他的方法解决这个问题。

     说明: 不要使用类的成员 / 全局 / 静态变量来存储状态，你的序列化和反序列化算法应该是无状态的。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func serialize(_ root: TreeNode?) -> String {
        if nil == root {
            return "X"
        }

        var res = [String]()
        var queue = [TreeNode]()
        queue.append(root!)
        while !queue.isEmpty {
            let node = queue.removeFirst()
            if Int.min != node.val {
                res.append(String(node.val))
                if let left = node.left {
                    queue.append(left)
                } else {
                    queue.append(TreeNode(Int.min))
                }
                if let right = node.right {
                    queue.append(right)
                } else {
                    queue.append(TreeNode(Int.min))
                }
            } else {
                res.append("X")
            }
        }
        return res.joined(separator: ",")
    }
    
    func deserialize(_ data: String) -> TreeNode? {
        if "X" == data {
            return nil
        }
        
        let arr = data.split(separator: ",")
        let first = arr[0]
        let root = TreeNode(Int(String(first))!)
        var queue = [TreeNode]()
        queue.append(root)
        var index = 1
        while index < arr.count {
            let node = queue.removeFirst()
            let left = arr[index]
            let right = arr[index + 1]
            if left != "X" {
                let leftNode = TreeNode(Int(String(left))!)
                node.left = leftNode
                queue.append(leftNode)
            }
            if right != "X" {
                let rightNode = TreeNode(Int(String(right))!)
                node.right = rightNode
                queue.append(rightNode)
            }
            index += 2
        }
        return root
    }
    
    /**
     236. 二叉树的最近公共祖先
     
     给定一个二叉树, 找到该树中两个指定节点的最近公共祖先。

     百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

     例如，给定如下二叉树:  root = [3,5,1,6,2,0,8,null,null,7,4]



      

     示例 1:

     输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
     输出: 3
     解释: 节点 5 和节点 1 的最近公共祖先是节点 3。
     示例 2:

     输入: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
     输出: 5
     解释: 节点 5 和节点 4 的最近公共祖先是节点 5。因为根据定义最近公共祖先节点可以为节点本身。
      

     说明:

     所有节点的值都是唯一的。
     p、q 为不同节点且均存在于给定的二叉树中。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // termimator
        if (null == root || p == root || q == root) return root;
        // drill down
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (null == left) return right;
        if (null == right) return left;
        return root;
    }
    
    /**
     106. 从中序与后序遍历序列构造二叉树
     
     根据一棵树的中序遍历与后序遍历构造二叉树。

     注意:
     你可以假设树中没有重复的元素。

     例如，给出

     中序遍历 inorder = [9,3,15,20,7]
     后序遍历 postorder = [9,15,7,20,3]
     返回如下的二叉树：

         3
        / \
       9  20
         /  \
        15   7

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func buildTree(_ inorder: [Int], _ postorder: [Int]) -> TreeNode? {
        if inorder.count > 0 && inorder.count == postorder.count {
            var dict = [Int : Int]()
            for i in 0 ... inorder.count - 1 {
                dict[inorder[i]] = i
            }
            let root = buildTreeHelper(inorder, 0, inorder.count - 1, postorder, 0, postorder.count - 1, dict)
            return root
        }
        
        return nil
    }
    
    private func buildTreeHelper(_ inorder: [Int], _ iStart: Int, _ iEnd: Int, _ postorder: [Int], _ pStart: Int, _ pEnd: Int, _ dict: [Int : Int]) -> TreeNode? {
        if pStart > pEnd || iStart > iEnd {
            return nil
        }
        
        let rootVal = postorder[pEnd]
        let iIndex = dict[rootVal]!
        let leftNodeNum = iIndex - iStart
        
        let root = TreeNode(rootVal)
        root.left = buildTreeHelper(inorder, iStart, iIndex - 1, postorder, pStart, pStart + leftNodeNum - 1, dict)
        root.right = buildTreeHelper(inorder, iIndex + 1, iEnd, postorder, pStart + leftNodeNum, pEnd - 1, dict)
        
        return root
    }
    
    /**
     77. 组合
     
     给定两个整数 n 和 k，返回 1 ... n 中所有可能的 k 个数的组合。

     示例:

     输入: n = 4, k = 2
     输出:
     [
       [2,4],
       [3,4],
       [2,3],
       [1,2],
       [1,3],
       [1,4],
     ]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/combinations
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private var combineRes = [[Int]]()
    func combine(_ n: Int, _ k: Int) -> [[Int]] {
        if n >= k {
            var nums = [Int]()
            for i in 1 ... n {
                nums.append(i)
            }
            var res = [Int]()
            combineBacktrack(0, k, nums, &res)
        }
        return combineRes
    }

    private func combineBacktrack(_ first: Int, _ k: Int, _ nums: [Int], _ res: inout [Int]) {
        if k == res.count {
            combineRes.append(res)
            return
        }

        for i in first ..< nums.count {
            res.append(nums[i])
            combineBacktrack(i + 1, k, nums, &res)
            res.removeLast()
        }
    }
    
    /**
     46. 全排列
     
     给定一个 没有重复 数字的序列，返回其所有可能的全排列。

     示例:

     输入: [1,2,3]
     输出:
     [
       [1,2,3],
       [1,3,2],
       [2,1,3],
       [2,3,1],
       [3,1,2],
       [3,2,1]
     ]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/permutations
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func permute(_ nums: [Int]) -> [[Int]] {
        var permuteRes = [[Int]]()
        var res = [Int]()
        permuteBacktrack(0, nums.count, nums, &res, &permuteRes)
        return permuteRes
    }
    
    private func permuteBacktrack(_ level: Int, _ count: Int, _ nums: [Int], _ res: inout [Int], _ finalRes: inout [[Int]]) {
        if res.count == count {
            finalRes.append(res)
            return
        }
        
        for i in 0 ... count - 1 {
            if res.contains(nums[i]) {
                continue
            }
            
            res.append(nums[i])
            permuteBacktrack(i + 1, count, nums, &res, &finalRes)
            res.removeLast()
        }
    }
    
    func permuteUnique(_ nums: [Int]) -> [[Int]] {
        var permuteUniqueRes = [[Int]]()
        var res = [Int]()
        var visit = [Bool](repeating: false, count: nums.count)
        permuteUniqueBacktrack(0, nums.count, nums.sorted(), &res, &visit, &permuteUniqueRes)
        return permuteUniqueRes
    }
    
    private func permuteUniqueBacktrack(_ level: Int, _ count: Int, _ nums: [Int], _ res: inout [Int], _ visit: inout [Bool], _ permuteUniqueRes: inout [[Int]]) {
        if res.count == count {
            permuteUniqueRes.append(res)
            return
        }
        
        for i in 0 ... count - 1 {
            if visit[i] || (i > 0 && nums[i] == nums[i - 1] && !visit[i - 1]) {
                continue
            }
            
            res.append(nums[i])
            visit[i] = true
            permuteUniqueBacktrack(i + 1, count, nums, &res, &visit, &permuteUniqueRes)
            visit[i] = false
            res.removeLast()
        }
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
    
    /**
     169. 多数元素
     
     给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。

     你可以假设数组是非空的，并且给定的数组总是存在多数元素。

      

     示例 1:

     输入: [3,2,3]
     输出: 3
     示例 2:

     输入: [2,2,1,1,1,2,2]
     输出: 2


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/majority-element
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func majorityElement(_ nums: [Int]) -> Int {
//        guard nums.count > 0 else {
//            return Int.min
//        }
//
//        var dict = [Int : Int]()
//        for num in nums {
//            if let count = dict[num] {
//                dict[num] = count + 1
//            } else {
//                dict[num] = 1
//            }
//        }
//
//        for key in dict.keys {
//            if dict[key]! > nums.count/2 {
//                return key
//            }
//        }
//
//        return Int.min
        
        // 摩尔投票法
        guard nums.count > 0 else {
            return Int.min
        }
        
        var majorNum = nums[0]
        var count = 1
        for i in 1 ..< nums.count {
            if majorNum == nums[i] {
                count += 1
            } else {
                count -= 1
                if 0 == count {
                    majorNum = nums[i]
                    count = 1
                }
            }
        }
        return majorNum
    }
    
    /**
     17. 电话号码的字母组合
     
     给定一个仅包含数字 2-9 的字符串，返回所有它能表示的字母组合。

     给出数字到字母的映射如下（与电话按键相同）。注意 1 不对应任何字母。



     示例:

     输入："23"
     输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
     说明:
     尽管上面的答案是按字典序排列的，但是你可以任意选择答案输出的顺序。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func letterCombinations(_ digits: String) -> [String] {
        var res = [String]()
        guard digits.count > 0 else {
            return res
        }
        
        let dict = ["2" : "abc", "3" : "def", "4" : "ghi", "5" : "jkl", "6" : "mno", "7" : "pqrs", "8" : "tuv", "9" : "wxyz"]
        var digitsArr = [String]()
        for str in digits {
            digitsArr.append(String(str))
        }
        if digitsArr.count > 0 {
            letterCombinateHelper(0, digitsArr, "", &res, dict)
        }
        return res
    }
    
    private func letterCombinateHelper(_ level: Int, _ digitsArr: [String], _ str: String, _ res: inout [String], _ dict: [String : String]) {
        if digitsArr.count == str.count {
            res.append(str)
            return
        }
        
        let digit = digitsArr[level]
        let digitStr = dict[digit]!
        let digitStrArr = Array(digitStr)
        for i in 0 ... digitStrArr.count - 1 {
            letterCombinateHelper(level + 1, digitsArr, str + String(digitStrArr[i]), &res, dict)
        }
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
     455. 分发饼干
     
     假设你是一位很棒的家长，想要给你的孩子们一些小饼干。但是，每个孩子最多只能给一块饼干。对每个孩子 i ，都有一个胃口值 gi ，这是能让孩子们满足胃口的饼干的最小尺寸；并且每块饼干 j ，都有一个尺寸 sj 。如果 sj >= gi ，我们可以将这个饼干 j 分配给孩子 i ，这个孩子会得到满足。你的目标是尽可能满足越多数量的孩子，并输出这个最大数值。

     注意：

     你可以假设胃口值为正。
     一个小朋友最多只能拥有一块饼干。

     示例 1:

     输入: [1,2,3], [1,1]

     输出: 1

     解释:
     你有三个孩子和两块小饼干，3个孩子的胃口值分别是：1,2,3。
     虽然你有两块小饼干，由于他们的尺寸都是1，你只能让胃口值是1的孩子满足。
     所以你应该输出1。
     示例 2:

     输入: [1,2], [1,2,3]

     输出: 2

     解释:
     你有两个孩子和三块小饼干，2个孩子的胃口值分别是1,2。
     你拥有的饼干数量和尺寸都足以让所有孩子满足。
     所以你应该输出2.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/assign-cookies
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findContentChildren(_ g: [Int], _ s: [Int]) -> Int {
        if 0 == s.count {
            return 0
        }
        
        var res = 0
        let sortedG = g.sorted()
        let sortedS = s.sorted()
        var i = 0
        for j in 0 ... sortedS.count - 1 {
            if i == sortedG.count {
                break
            }
            
            if sortedS[j] >= sortedG[i] {
                res += 1
                i += 1
            }
        }
        return res
    }
    
    /**
     55. 跳跃游戏
     
     给定一个非负整数数组，你最初位于数组的第一个位置。

     数组中的每个元素代表你在该位置可以跳跃的最大长度。

     判断你是否能够到达最后一个位置。

     示例 1:

     输入: [2,3,1,1,4]
     输出: true
     解释: 我们可以先跳 1 步，从位置 0 到达 位置 1, 然后再从位置 1 跳 3 步到达最后一个位置。
     示例 2:

     输入: [3,2,1,0,4]
     输出: false
     解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/jump-game
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func canJump(_ nums: [Int]) -> Bool {
        let n = nums.count
        var rightmost = 0
        for i in 0 ... n - 1 {
            if i <= rightmost {
                rightmost = max(rightmost, nums[i] + i)
                if rightmost >= n - 1 {
                    return true
                }
            }
        }
        return false
    }
    
    /**
     122. 买卖股票的最佳时机 II
     
     给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

     设计一个算法来计算你所能获取的最大利润。你可以尽可能地完成更多的交易（多次买卖一支股票）。

     注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

      

     示例 1:

     输入: [7,1,5,3,6,4]
     输出: 7
     解释: 在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
          随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6-3 = 3 。
     示例 2:

     输入: [1,2,3,4,5]
     输出: 4
     解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
          注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
          因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
     示例 3:

     输入: [7,6,4,3,1]
     输出: 0
     解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit(_ prices: [Int]) -> Int {
        var res = 0
        if prices.count > 1 {
            for i in 1 ... prices.count - 1 {
                let tmp = prices[i] - prices[i - 1]
                if tmp > 0 {
                    res += tmp
                }
            }
        }
        return res
    }
    
    /**
     69. x 的平方根
     
     实现 int sqrt(int x) 函数。

     计算并返回 x 的平方根，其中 x 是非负整数。

     由于返回类型是整数，结果只保留整数的部分，小数部分将被舍去。

     示例 1:

     输入: 4
     输出: 2
     示例 2:

     输入: 8
     输出: 2
     说明: 8 的平方根是 2.82842...,
          由于返回类型是整数，小数部分将被舍去。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sqrtx
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func mySqrt(_ x: Int) -> Int {
        if x < 2 {
            return x
        }
        
        var left: Double = 0
        var right: Double = Double(x)
        var middle = (left + right)/2.0
        var product = middle * middle
        while fabs(product - Double(x)) > 0.000001 {
            if product > Double(x) {
                right = middle
                middle = (left + right)/2.0
            } else {
                left = middle
                middle = (left + right)/2.0
            }
            product = middle * middle
        }
        
        return Int(floor(middle))
    }
    
    /**
     367. 有效的完全平方数
     
     给定一个正整数 num，编写一个函数，如果 num 是一个完全平方数，则返回 True，否则返回 False。

     说明：不要使用任何内置的库函数，如  sqrt。

     示例 1：

     输入：16
     输出：True
     示例 2：

     输入：14
     输出：False

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-perfect-square
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isPerfectSquare(_ num: Int) -> Bool {
        if num < 2 {
            return true
        }
        
        var left = 2
        var right = num
        var middle = (left + right)/2
        while left < right {
            if num == middle * middle {
                return true
            } else if num > middle * middle {
                left = middle
                middle = (left + right)/2
                if left == middle {
                    break
                }
            } else {
                right = middle
                middle = (left + right)/2
                if right == middle {
                    break
                }
            }
        }
        
        return false
    }
}
