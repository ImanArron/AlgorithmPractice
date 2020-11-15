//
//  Homework_week_09.swift
//  DataStructDemo
//
//  Created by noctis on 2020/11/9.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

class Homework_week_09 {
    /**
     973. 最接近原点的 K 个点
     
     我们有一个由平面上的点组成的列表 points。需要从中找出 K 个距离原点 (0, 0) 最近的点。

     （这里，平面上两点之间的距离是欧几里德距离。）

     你可以按任何顺序返回答案。除了点坐标的顺序之外，答案确保是唯一的。

      

     示例 1：

     输入：points = [[1,3],[-2,2]], K = 1
     输出：[[-2,2]]
     解释：
     (1, 3) 和原点之间的距离为 sqrt(10)，
     (-2, 2) 和原点之间的距离为 sqrt(8)，
     由于 sqrt(8) < sqrt(10)，(-2, 2) 离原点更近。
     我们只需要距离原点最近的 K = 1 个点，所以答案就是 [[-2,2]]。
     示例 2：

     输入：points = [[3,3],[5,-1],[-2,4]], K = 2
     输出：[[3,3],[-2,4]]
     （答案 [[-2,4],[3,3]] 也会被接受。）
      

     提示：

     1 <= K <= points.length <= 10000
     -10000 < points[i][0] < 10000
     -10000 < points[i][1] < 10000

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/k-closest-points-to-origin
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func kClosest(_ points: [[Int]], _ K: Int) -> [[Int]] {
        // 方法一：使用系统库函数
//        let sortedPoints = points.sorted { (a, b) -> Bool in
//            return a[0] * a[0] + a[1] * a[1] < b[0] * b[0] + b[1] * b[1]
//        }
//
//        var res = [[Int]]()
//        for i in 0 ..< K {
//            if i < sortedPoints.count {
//                res.append(sortedPoints[i])
//            } else {
//                break
//            }
//        }
//
//        return res
        
        // 方法二：快排
        if K >= points.count {
            return points
        }
        
        let n = points.count
        var sortedPoints = points
        var index = partition(&sortedPoints, 0, n - 1)
        while index != K {
            if index > K {
                index = partition(&sortedPoints, 0, index - 1)
            } else if index < K {
                index = partition(&sortedPoints, index + 1, n - 1)
            }
        }
        
        return Array(sortedPoints[0 ..< K])
    }
    
    private func partition(_ points: inout [[Int]], _ begin: Int, _ end: Int) -> Int {
        if begin == end {
            return -1
        }
        
        let pivot = end
        var count = begin
        for i in begin ..< end {
            let p1 = points[i]
            let p2 = points[pivot]
            if p1[0] * p1[0] + p1[1] * p1[1] < p2[0] * p2[0] + p2[1] * p2[1] {
                let temp = points[count]
                points[count] = p1
                points[i] = temp
                count += 1
            }
        }
        
        let temp = points[count]
        points[count] = points[pivot]
        points[pivot] = temp
        return count
    }
    
    /**
     83. 删除排序链表中的重复元素
     
     给定一个排序链表，删除所有重复的元素，使得每个元素只出现一次。

     示例 1:

     输入: 1->1->2
     输出: 1->2
     示例 2:

     输入: 1->1->2->3->3
     输出: 1->2->3

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func deleteDuplicates(_ head: ListNode?) -> ListNode? {
        let dummy = ListNode(0)
        dummy.next = head
        var node = head
        while nil != node {
            var next = node!.next
            while nil != next && node!.val == next!.val {
                node!.next = next!.next
                next = next!.next
            }
            node = next
        }
        return dummy.next
    }
    
    /**
     70. 爬楼梯
     
     假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

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
     */
    func climbStairs(_ n: Int) -> Int {
        if n < 3 {
            return n
        }
        
        var f1 = 1
        var f2 = 2
        var res = 0
        for _ in 3 ... n {
            res = f1 + f2
            f1 = f2
            f2 = res
        }
        return res
    }
    
    /**
     62. 不同路径
     
     一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

     机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

     问总共有多少条不同的路径？



     例如，上图是一个7 x 3 的网格。有多少可能的路径？

      

     示例 1:

     输入: m = 3, n = 2
     输出: 3
     解释:
     从左上角开始，总共有 3 条路径可以到达右下角。
     1. 向右 -> 向右 -> 向下
     2. 向右 -> 向下 -> 向右
     3. 向下 -> 向右 -> 向右
     示例 2:

     输入: m = 7, n = 3
     输出: 28
      

     提示：

     1 <= m, n <= 100
     题目数据保证答案小于等于 2 * 10 ^ 9


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/unique-paths
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func uniquePaths(_ m: Int, _ n: Int) -> Int {
        guard m > 0 && n > 0 else {
            return 0
        }
        
        var dp = [[Int]]()
        for _ in 0 ..< m {
            dp.append([Int](repeating: 1, count: n))
        }
        
        for i in (0 ..< m - 1).reversed() {
            for j in (0 ..< n - 1).reversed() {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1]
            }
        }
        
        return dp[0][0]
    }
    
    /**
     198. 打家劫舍
     
     你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

     给定一个代表每个房屋存放金额的非负整数数组，计算你 不触动警报装置的情况下 ，一夜之内能够偷窃到的最高金额。

      

     示例 1：

     输入：[1,2,3,1]
     输出：4
     解释：偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
          偷窃到的最高金额 = 1 + 3 = 4 。
     示例 2：

     输入：[2,7,9,3,1]
     输出：12
     解释：偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。
          偷窃到的最高金额 = 2 + 9 + 1 = 12 。
      

     提示：

     0 <= nums.length <= 100
     0 <= nums[i] <= 400

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/house-robber
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func rob(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 0 else {
            return 0
        }
        
        var dp = [Int](repeating: 0, count: n)
        dp[0] = nums[0]
        for i in 1 ..< n {
            dp[i] = i > 1 ? max(dp[i - 1], dp[i - 2] + nums[i]) : max(nums[0], nums[1])
        }
        return dp[n - 1]
    }
    
    /**
     64. 最小路径和
     
     给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

     说明：每次只能向下或者向右移动一步。

     示例:

     输入:
     [
       [1,3,1],
       [1,5,1],
       [4,2,1]
     ]
     输出: 7
     解释: 因为路径 1→3→1→1→1 的总和最小。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/minimum-path-sum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minPathSum(_ grid: [[Int]]) -> Int {
        let m = grid.count
        guard m > 0 else {
            return 0
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return 0
        }
        
        var dp = [[Int]]()
        for _ in 0 ..< m {
            dp.append([Int](repeating: 0, count: n))
        }
        
        dp[0][0] = grid[0][0]
        for i in 1 ..< m {
            dp[i][0] = dp[i - 1][0] + grid[i][0]
        }
        for i in 1 ..< n {
            dp[0][i] = dp[0][i - 1] + grid[0][i]
        }
        for i in 1 ..< m {
            for j in 1 ..< n {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
            }
        }
        return dp[m - 1][n - 1]
    }
    
    /**
     121. 买卖股票的最佳时机
     
     给定一个数组，它的第 i 个元素是一支给定股票第 i 天的价格。

     如果你最多只允许完成一笔交易（即买入和卖出一支股票一次），设计一个算法来计算你所能获取的最大利润。

     注意：你不能在买入股票前卖出股票。

      

     示例 1:

     输入: [7,1,5,3,6,4]
     输出: 5
     解释: 在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
          注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。
     示例 2:

     输入: [7,6,4,3,1]
     输出: 0
     解释: 在这种情况下, 没有交易完成, 所以最大利润为 0。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit(_ prices: [Int]) -> Int {
        let n = prices.count
        guard n > 1 else {
            return 0
        }

        var dp = [Int](repeating: 0, count: n)
        var minPrice = prices[0]
        for i in 1 ..< n {
            minPrice = min(minPrice, prices[i])
            if prices[i] > minPrice {
                dp[i] = prices[i] - minPrice
            }
        }

        var res = 0
        for num in dp {
            res = max(res, num)
        }
        return res
    }
    
    /**
     746. 使用最小花费爬楼梯
     
     数组的每个索引作为一个阶梯，第 i个阶梯对应着一个非负数的体力花费值 cost[i](索引从0开始)。

     每当你爬上一个阶梯你都要花费对应的体力花费值，然后你可以选择继续爬一个阶梯或者爬两个阶梯。

     您需要找到达到楼层顶部的最低花费。在开始时，你可以选择从索引为 0 或 1 的元素作为初始阶梯。

     示例 1:

     输入: cost = [10, 15, 20]
     输出: 15
     解释: 最低花费是从cost[1]开始，然后走两步即可到阶梯顶，一共花费15。
      示例 2:

     输入: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
     输出: 6
     解释: 最低花费方式是从cost[0]开始，逐个经过那些1，跳过cost[3]，一共花费6。
     注意：

     cost 的长度将会在 [2, 1000]。
     每一个 cost[i] 将会是一个Integer类型，范围为 [0, 999]。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/min-cost-climbing-stairs
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minCostClimbingStairs(_ cost: [Int]) -> Int {
        var f1 = 0, f2 = 0
        for i in (0 ..< cost.count).reversed() {
            let f0 = cost[i] + min(f1, f2)
            f2 = f1
            f1 = f0
        }
        return min(f1, f2)
    }
    
    /**
     300. 最长上升子序列
     
     给定一个无序的整数数组，找到其中最长上升子序列的长度。

     示例:

     输入: [10,9,2,5,3,7,101,18]
     输出: 4
     解释: 最长的上升子序列是 [2,3,7,101]，它的长度是 4。
     说明:

     可能会有多种最长上升子序列的组合，你只需要输出对应的长度即可。
     你算法的时间复杂度应该为 O(n2) 。
     进阶: 你能将算法的时间复杂度降低到 O(n log n) 吗?



     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/longest-increasing-subsequence
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func lengthOfLIS(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 1 else {
            return n
        }
        
        var dp = [Int](repeating: 1, count: n)
        var len = 0
        for i in 1 ..< n {
            for j in 0 ..< i {
                if nums[i] > nums[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
            len = max(len, dp[i])
        }
        return len
    }
    
    /**
     91. 解码方法
     
     一条包含字母 A-Z 的消息通过以下方式进行了编码：

     'A' -> 1
     'B' -> 2
     ...
     'Z' -> 26
     给定一个只包含数字的非空字符串，请计算解码方法的总数。

     题目数据保证答案肯定是一个 32 位的整数。

      

     示例 1：

     输入："12"
     输出：2
     解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
     示例 2：

     输入："226"
     输出：3
     解释：它可以解码为 "BZ" (2 26), "VF" (22 6), 或者 "BBF" (2 2 6) 。
     示例 3：

     输入：s = "0"
     输出：0
     示例 4：

     输入：s = "1"
     输出：1
     示例 5：

     输入：s = "2"
     输出：1
      

     提示：

     1 <= s.length <= 100
     s 只包含数字，并且可以包含前导零。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/decode-ways
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numDecodings(_ s: String) -> Int {
        let sArr = Array(s)
        guard sArr.count > 0 else {
            return 0
        }
        
        let zero: Character = "0"
        guard zero != sArr[0] else {
            return 0
        }
        
        var dp = [Int](repeating: 0, count: sArr.count)
        dp[0] = 1
        for i in 1 ..< sArr.count {
            if zero != sArr[i] {
                dp[i] = dp[i - 1]
            }
            let num = 10 * (sArr[i - 1].asciiValue! - zero.asciiValue!) + (sArr[i].asciiValue! - zero.asciiValue!)
            if num >= 10 && num <= 26 {
                if 1 == i {
                    dp[i] = dp[i] + 1
                } else {
                    dp[i] = dp[i] + dp[i - 2]
                }
            }
        }
        return dp[sArr.count - 1]
    }
    
    /**
     120. 三角形最小路径和
     
     给定一个三角形，找出自顶向下的最小路径和。每一步只能移动到下一行中相邻的结点上。

     相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。

      

     例如，给定三角形：

     [
          [2],
         [3,4],
        [6,5,7],
       [4,1,8,3]
     ]
     自顶向下的最小路径和为 11（即，2 + 3 + 5 + 1 = 11）。

      

     说明：

     如果你可以只使用 O(n) 的额外空间（n 为三角形的总行数）来解决这个问题，那么你的算法会很加分。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/triangle
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        var dp = triangle
        let m = triangle.count
        guard m > 0 else {
            return 0
        }
        
        for i in (0 ..< m - 1).reversed() {
            let nums = triangle[i]
            for j in 0 ..< nums.count {
                dp[i][j] += min(dp[i + 1][j], dp[i + 1][j + 1])
            }
        }
        
        return dp[0][0]
    }
    
    /**
     85. 最大矩形
     
     给定一个仅包含 0 和 1 、大小为 rows x cols 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

      

     示例 1：


     输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
     输出：6
     解释：最大矩形如上图所示。
     示例 2：

     输入：matrix = []
     输出：0
     示例 3：

     输入：matrix = [["0"]]
     输出：0
     示例 4：

     输入：matrix = [["1"]]
     输出：1
     示例 5：

     输入：matrix = [["0","0"]]
     输出：0
      

     提示：

     rows == matrix.length
     cols == matrix.length
     0 <= row, cols <= 200
     matrix[i][j] 为 '0' 或 '1'


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximal-rectangle
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maximalRectangle(_ matrix: [[Character]]) -> Int {
        /**
         解题思路
         动态规划

         heights[i][j]代表[i，j]的高度
         heights[i][j] = matrix[i][j]=='1'? heights[i-1][j] + 1:0

         dp[i][j][k] 代表以[i,j]为右下角，高度为k可以组成的面积
         dp[i][j][k] = dp[i][j-1][k] + k

         作者：leeyupeng
         链接：https://leetcode-cn.com/problems/maximal-rectangle/solution/dong-tai-gui-hua-by-leeyupeng-5/
         来源：力扣（LeetCode）
         著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
         */
        let m = matrix.count
        guard m > 0 else {
            return 0
        }
        let n = matrix[0].count
        guard n > 0 else {
            return 0
        }
        
        var heights = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        var dp = [[[Int]]](repeating: [[Int]](repeating: [Int](repeating: 0, count: m + 1), count: n + 1), count: m + 1)
        var res = 0
        for i in 1 ... m {
            for j in 1 ... n {
                if "0" == matrix[i - 1][j - 1] {
                    continue
                }
                
                heights[i][j] = heights[i - 1][j] + 1
                for k in 1 ... heights[i][j] {
                    dp[i][j][k] = dp[i][j - 1][k] + k
                    res = max(res, dp[i][j][k])
                }
            }
        }
        return res
    }
    
    /**
     31. 下一个排列
     
     实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。

     如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。

     必须原地修改，只允许使用额外常数空间。

     以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
     1,2,3 → 1,3,2
     3,2,1 → 1,2,3
     1,1,5 → 1,5,1

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/next-permutation
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func nextPermutation(_ nums: inout [Int]) {
        /**
         1 先找出最大的索引 k 满足 nums[k] < nums[k+1]，如果不存在，就翻转整个数组；
         2 再找出另一个最大索引 l 满足 nums[l] > nums[k]；
         3 交换 nums[l] 和 nums[k]；
         4 最后翻转 nums[k+1:]

         作者：powcai
         链接：https://leetcode-cn.com/problems/next-permutation/solution/xia-yi-ge-pai-lie-by-powcai/
         来源：力扣（LeetCode）
         著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
         */
        guard nums.count > 1 else {
            return
        }
        
        var k = -1
        for i in (0 ..< nums.count - 1).reversed() {
            if nums[i] < nums[i + 1] {
                k = i
                break
            }
        }
        
        if -1 == k {
            reverseNums(&nums, 0, nums.count - 1)
            return
        }
        
        var l = -1
        for i in (0 ..< nums.count).reversed() {
            if nums[i] > nums[k] {
                l = i
                break
            }
        }
        
        let temp = nums[k]
        nums[k] = nums[l]
        nums[l] = temp
        
        reverseNums(&nums, k + 1, nums.count - 1)
    }
    
    private func reverseNums(_ nums: inout [Int], _ l: Int, _ r: Int) {
        var p1 = l, p2 = r
        while p1 < p2 {
            let temp = nums[p1]
            nums[p1] = nums[p2]
            nums[p2] = temp
            p1 += 1
            p2 -= 1
        }
    }
    
    /**
     127. 单词接龙
     
     给定两个单词（beginWord 和 endWord）和一个字典，找到从 beginWord 到 endWord 的最短转换序列的长度。转换需遵循如下规则：

     每次转换只能改变一个字母。
     转换过程中的中间单词必须是字典中的单词。
     说明:

     如果不存在这样的转换序列，返回 0。
     所有单词具有相同的长度。
     所有单词只由小写字母组成。
     字典中不存在重复的单词。
     你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
     示例 1:

     输入:
     beginWord = "hit",
     endWord = "cog",
     wordList = ["hot","dot","dog","lot","log","cog"]

     输出: 5

     解释: 一个最短转换序列是 "hit" -> "hot" -> "dot" -> "dog" -> "cog",
          返回它的长度 5。
     示例 2:

     输入:
     beginWord = "hit"
     endWord = "cog"
     wordList = ["hot","dot","dog","lot","log"]

     输出: 0

     解释: endWord "cog" 不在字典中，所以无法进行转换。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/word-ladder
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func ladderLength(_ beginWord: String, _ endWord: String, _ wordList: [String]) -> Int {
        let wordSet = Set(wordList)
        guard wordSet.contains(endWord) else {
            return 0
        }
        
        var visited = Set<String>(), beginSet = Set<String>(), endSet = Set<String>()
        beginSet.insert(beginWord)
        endSet.insert(endWord)
        visited.insert(beginWord)
        visited.insert(endWord)
        
        var res = 1
        let letters: [Character] = Array("abcdefghijklmnopqrstuvwxyz")
        while !beginSet.isEmpty && !endSet.isEmpty {
            if beginSet.count > endSet.count {
                let temp = beginSet
                beginSet = endSet
                endSet = temp
            }
            
            var tempSet = Set<String>()
            for word in beginSet {
                var wordArr = Array(word)
                for i in 0 ..< wordArr.count {
                    let char = wordArr[i]
                    for letter in letters {
                        if char == letter {
                            continue
                        }
                        
                        wordArr[i] = letter
                        var nextWord = ""
                        for j in 0 ..< wordArr.count {
                            nextWord.append(wordArr[j])
                        }
                        
                        if endSet.contains(nextWord) {
                            return res + 1
                        }
                        
                        if wordSet.contains(nextWord) && !visited.contains(nextWord) {
                            tempSet.insert(nextWord)
                            visited.insert(nextWord)
                        }
                    }
                    wordArr[i] = char
                }
            }
            
            res += 1
            beginSet = tempSet
        }
        return 0
    }
    
    /**
     32. 最长有效括号
     
     给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

     示例 1:

     输入: "(()"
     输出: 2
     解释: 最长有效括号子串为 "()"
     示例 2:

     输入: ")()())"
     输出: 4
     解释: 最长有效括号子串为 "()()"

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/longest-valid-parentheses
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func longestValidParentheses(_ s: String) -> Int {
        let sArr = Array(s)
        guard sArr.count > 1 else {
            return 0
        }
        
        var res = 0
        // dp[i] 表示 sArr[0 ..< i] 的最长有效括号长度
        var dp = [Int](repeating: 0, count: sArr.count)
        for i in 1 ..< sArr.count {
            if ")" == sArr[i] {
                if "(" == sArr[i - 1] {
                    dp[i] = 1 == i ? 2 : dp[i - 2] + 2
                } else if i > dp[i - 1] && "(" == sArr[i - dp[i - 1] - 1] {
                    dp[i] = i - dp[i - 1] > 1 ? dp[i - 1] + dp[i - dp[i - 1] - 2] + 2 : dp[i - 1] + 2
                }
                res = max(res, dp[i])
            }
        }
        return res
    }
    
    /**
     115. 不同的子序列
     
     给定一个字符串 s 和一个字符串 t ，计算在 s 的子序列中 t 出现的个数。

     字符串的一个 子序列 是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）

     题目数据保证答案符合 32 位带符号整数范围。

      

     示例 1：

     输入：s = "rabbbit", t = "rabbit"
     输出：3
     解释：
     如下图所示, 有 3 种可以从 s 中得到 "rabbit" 的方案。
     (上箭头符号 ^ 表示选取的字母)
     rabbbit
     ^^^^ ^^
     rabbbit
     ^^ ^^^^
     rabbbit
     ^^^ ^^^
     示例 2：

     输入：s = "babgbag", t = "bag"
     输出：5
     解释：
     如下图所示, 有 5 种可以从 s 中得到 "bag" 的方案。
     (上箭头符号 ^ 表示选取的字母)
     babgbag
     ^^ ^
     babgbag
     ^^    ^
     babgbag
     ^    ^^
     babgbag
       ^  ^^
     babgbag
         ^^^
      

     提示：

     0 <= s.length, t.length <= 1000
     s 和 t 由英文字母组成

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/distinct-subsequences
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numDistinct(_ s: String, _ t: String) -> Int {
        guard s.count > t.count else {
            return s == t ? 1 : 0
        }
        
        let tArr = Array(t)
        let sArr = Array(s)
        let m = tArr.count
        let n = sArr.count
        /**
         dp[i][j] 代表 T 前 i 字符串可以由 S j 字符串组成最多个数.

         动态方程:

         当 S[j] == T[i] , dp[i][j] = dp[i-1][j-1] + dp[i][j-1];
         当 S[j] != T[i] , dp[i][j] = dp[i][j-1]
         */
        var dp = [[Int]](repeating: [Int](repeating: 0, count: n + 1), count: m + 1)
        for i in 0 ... n {
            dp[0][i] = 1
        }
        for i in 1 ... m {
            for j in 1 ... n {
                if tArr[i - 1] == sArr[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1]
                } else {
                    dp[i][j] = dp[i][j - 1]
                }
            }
        }
        return dp[m][n]
        
//        let sArr = Array(s)
//        var res = 0
//        var tempS = ""
//        dfsNumDistinct(sArr, &tempS, t, &res)
//        return res
    }
    
    private func dfsNumDistinct(_ sArr: [Character], _ s: inout String, _ t: String, _ res: inout Int) {
        if s == t {
            res += 1
            return
        }
        
        if s.count == sArr.count {
            return
        }
        
        for i in 0 ..< sArr.count {
            s.append(sArr[i])
            dfsNumDistinct(sArr, &s, t, &res)
            s.removeLast()
        }
    }
    
    /**
     818. 赛车
     
     你的赛车起始停留在位置 0，速度为 +1，正行驶在一个无限长的数轴上。（车也可以向负数方向行驶。）

     你的车会根据一系列由 A（加速）和 R（倒车）组成的指令进行自动驾驶 。

     当车得到指令 "A" 时, 将会做出以下操作： position += speed, speed *= 2。

     当车得到指令 "R" 时, 将会做出以下操作：如果当前速度是正数，则将车速调整为 speed = -1 ；否则将车速调整为 speed = 1。  (当前所处位置不变。)

     例如，当得到一系列指令 "AAR" 后, 你的车将会走过位置 0->1->3->3，并且速度变化为 1->2->4->-1。

     现在给定一个目标位置，请给出能够到达目标位置的最短指令列表的长度。

     示例 1:
     输入:
     target = 3
     输出: 2
     解释:
     最短指令列表为 "AA"
     位置变化为 0->1->3
     示例 2:
     输入:
     target = 6
     输出: 5
     解释:
     最短指令列表为 "AAARA"
     位置变化为 0->1->3->7->7->6
     说明:

     1 <= target（目标位置） <= 10000。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/race-car
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func racecar(_ target: Int) -> Int {
        var dp = [Int](repeating: 0, count: 10001)
        return racecarRecusive(target, &dp)
    }
    
    private func racecarRecusive(_ target: Int, _ dp: inout [Int]) -> Int {
        if dp[target] > 0 {
            return dp[target]
        }
        
        let n = Int(floor(log2(Double(target)))) + 1
        if target + 1 == (1 << n) {
            dp[target] = n
        } else {
            // n个A到达2^n-1位置，然后R反向，走完剩余
            dp[target] = n + 1 + racecarRecusive((1 << n) - 1 - target, &dp)
            // n-1个A到达2^(n-1)-1位置，然后R反向走m个A，再R反向，走完剩余
            // m取值遍历[0, n-1)
            for i in 0 ..< n - 1 {
                dp[target] = min(dp[target], n + i + 1 + racecarRecusive(target - (1 << (n - 1)) + (1 << i), &dp))
            }
        }
        return dp[target]
    }
    
    private func numberOfLeadingZeros(_ x: Int) -> Int {
        var i = x
        if 0 == i {
            return 32
        }
        
        var n = 1
        if i >> 16 == 0 {
            n += 16
            i = i << 16
        }
        if i >> 24 == 0 {
            n += 8
            i = i << 8
        }
        if i >> 28 == 0 {
            n += 4
            i = i << 4
        }
        if i >> 30 == 0 {
            n += 2
            i = i << 2
        }
        n -= i >> 31
        return n
    }
    
    /**
     922. 按奇偶排序数组 II
     
     给定一个非负整数数组 A， A 中一半整数是奇数，一半整数是偶数。

     对数组进行排序，以便当 A[i] 为奇数时，i 也是奇数；当 A[i] 为偶数时， i 也是偶数。

     你可以返回任何满足上述条件的数组作为答案。

      

     示例：

     输入：[4,2,5,7]
     输出：[4,5,2,7]
     解释：[4,7,2,5]，[2,5,4,7]，[2,7,4,5] 也会被接受。
      

     提示：

     2 <= A.length <= 20000
     A.length % 2 == 0
     0 <= A[i] <= 1000

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sort-array-by-parity-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func sortArrayByParityII(_ A: [Int]) -> [Int] {
//        var res = A
//        for i in 0 ..< A.count {
//            if (0 == (i & 1) && 0 == (res[i] & 1)) || (0 != (i & 1) && 0 != (res[i] & 1)) {
//                continue
//            } else {
//                var j = i + 1
//                while j < A.count {
//                    if (0 == (i & 1) && 0 == (res[j] & 1)) || (0 != (i & 1) && 0 != (res[j] & 1)) {
//                        let temp = res[i]
//                        res[i] = res[j]
//                        res[j] = temp
//                        break
//                    }
//                    j += 1
//                }
//            }
//        }
//        return res
        
        // 桶排序
        var counts = [Int](repeating: 0, count: 1001)
        for i in 0 ..< A.count {
            counts[A[i]] += 1
        }
        
        var res = A
        var evenIdx = 0
        var oddIdx = 1
        for i in 0 ..< counts.count {
            var count = counts[i]
            while count > 0 {
                if 0 == (i & 1) {
                    res[evenIdx] = i
                    evenIdx += 2
                } else {
                    res[oddIdx] = i
                    oddIdx += 2
                }
                count -= 1
            }
        }
        return res
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
        guard nums.count > 0 else {
            return [[Int]]()
        }
        
        var res = [[Int]]()
        var temp = [Int]()
        var set = Set<Int>()
        permuteBacktrack(nums, &temp, &set, &res)
        return res
    }
    
    private func permuteBacktrack(_ nums: [Int], _ permute: inout [Int], _ set: inout Set<Int>, _ res: inout [[Int]]) {
        if nums.count == permute.count {
            res.append(permute)
            return
        }
        
        for i in 0 ..< nums.count {
            if set.contains(nums[i]) {
                continue
            }
            
            permute.append(nums[i])
            set.insert(nums[i])
            permuteBacktrack(nums, &permute, &set, &res)
            permute.removeLast()
            set.remove(nums[i])
        }
    }
    
    /**
     709. 转换成小写字母
     
     实现函数 ToLowerCase()，该函数接收一个字符串参数 str，并将该字符串中的大写字母转换成小写字母，之后返回新的字符串。

      

     示例 1：

     输入: "Hello"
     输出: "hello"
     示例 2：

     输入: "here"
     输出: "here"
     示例 3：

     输入: "LOVELY"
     输出: "lovely"

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/to-lower-case
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func toLowerCase(_ str: String) -> String {
//        let dict: [Character: Character] = ["A": "a", "B": "b", "C": "c", "D": "d", "E": "e", "F": "f", "G": "g", "H": "h", "I": "i", "J": "j", "K": "k", "L": "l", "M": "m", "N": "n", "O": "o", "P": "p", "Q": "q", "R": "r", "S": "s", "T": "t", "U": "u", "V": "v", "W": "w", "X": "x", "Y": "y", "Z": "z"]
//        var res = ""
//        for char in Array(str) {
//            if let lower = dict[char] {
//                res.append(lower)
//            } else {
//                res.append(char)
//            }
//        }
//        return res
         
        /**
         位运算解法
         
         1 大写变小写、小写变大写：字符 ^= 32;
         2 大写变小写、小写变小写：字符 |= 32;
         3 大写变大写、小写变大写：字符 &= 33;
         */
        var res = ""
        for c in Array(str) {
            let val = c.asciiValue! | 32
            res = res.appendingFormat("%c", val)
        }
        return res
    }
    
    /**
     58. 最后一个单词的长度
     
     给定一个仅包含大小写字母和空格 ' ' 的字符串 s，返回其最后一个单词的长度。如果字符串从左向右滚动显示，那么最后一个单词就是最后出现的单词。

     如果不存在最后一个单词，请返回 0 。

     说明：一个单词是指仅由字母组成、不包含任何空格字符的 最大子字符串。

      

     示例:

     输入: "Hello World"
     输出: 5

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/length-of-last-word
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func lengthOfLastWord(_ s: String) -> Int {
        guard s.count > 0 else {
            return 0
        }
        
        let sArr = Array(s)
        var end = sArr.count - 1
        while end >= 0 && " " == sArr[end] {
            end -= 1
        }
        
        if end < 0 {
            return 0
        }
        
        var start = end
        while start >= 0 && " " != sArr[start] {
            start -= 1
        }
        
        return end - start
    }
    
    /**
     771. 宝石与石头
     
     给定字符串J 代表石头中宝石的类型，和字符串 S代表你拥有的石头。 S 中每个字符代表了一种你拥有的石头的类型，你想知道你拥有的石头中有多少是宝石。

     J 中的字母不重复，J 和 S中的所有字符都是字母。字母区分大小写，因此"a"和"A"是不同类型的石头。

     示例 1:

     输入: J = "aA", S = "aAAbbbb"
     输出: 3
     示例 2:

     输入: J = "z", S = "ZZ"
     输出: 0
     注意:

     S 和 J 最多含有50个字母。
     J 中的字符不重复。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/jewels-and-stones
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numJewelsInStones(_ J: String, _ S: String) -> Int {
        guard J.count > 0 && S.count > 0 else {
            return 0
        }
        
        let jSet = Set(Array(J))
        let sArr = Array(S)
        var res = 0
        for c in sArr {
            if jSet.contains(c) {
                res += 1
            }
        }
        return res
    }
    
    /**
     8. 字符串转换整数 (atoi)
     
     请你来实现一个 atoi 函数，使其能将字符串转换成整数。

     首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。接下来的转化规则如下：

     如果第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字字符组合起来，形成一个有符号整数。
     假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成一个整数。
     该字符串在有效的整数部分之后也可能会存在多余的字符，那么这些字符可以被忽略，它们对函数不应该造成影响。
     注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换，即无法进行有效转换。

     在任何情况下，若函数不能进行有效的转换时，请返回 0 。

     提示：

     本题中的空白字符只包括空格字符 ' ' 。
     假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，请返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
      

     示例 1:

     输入: "42"
     输出: 42
     示例 2:

     输入: "   -42"
     输出: -42
     解释: 第一个非空白字符为 '-', 它是一个负号。
          我们尽可能将负号与后面所有连续出现的数字组合起来，最后得到 -42 。
     示例 3:

     输入: "4193 with words"
     输出: 4193
     解释: 转换截止于数字 '3' ，因为它的下一个字符不为数字。
     示例 4:

     输入: "words and 987"
     输出: 0
     解释: 第一个非空字符是 'w', 但它不是数字或正、负号。
          因此无法执行有效的转换。
     示例 5:

     输入: "-91283472332"
     输出: -2147483648
     解释: 数字 "-91283472332" 超过 32 位有符号整数范围。
          因此返回 INT_MIN (−231) 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/string-to-integer-atoi
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func myAtoi(_ s: String) -> Int {
        var sArr = Array(s)
        // 去掉开头的空格
        while sArr.count > 0 && " " == sArr[0] {
            sArr.removeFirst()
        }
        var numStr = ""
        var isNegative = false
        for i in 0 ..< sArr.count {
            let asciiValue = sArr[i].asciiValue!
            if 0 == i {
                // 第一位，只能为 1 - 9、-、+
                if (asciiValue >= 48 && asciiValue <= 57) || 45 == asciiValue || 43 == asciiValue {
                    if 45 == asciiValue {
                        isNegative = true
                    } else if asciiValue >= 48 && asciiValue <= 57 {
                        if 0 == numStr.count && 48 == asciiValue {
                            continue
                        }
                        numStr = numStr.appendingFormat("%c", asciiValue)
                    }
                    continue
                } else {
                    return 0
                }
            }
                        
            if asciiValue >= 48 && asciiValue <= 57 {
                if 0 == numStr.count && 48 == asciiValue {
                    continue
                }
                numStr = numStr.appendingFormat("%c", asciiValue)
            } else {
                break
            }
        }
        
        if 0 == numStr.count {
            return 0
        }
        
        if numStr.count > 10 {
            return isNegative ? -2147483648 : 2147483647
        }
        
        var res = isNegative ? -Int(numStr)! : Int(numStr)!
        if res > 2147483647 {
            res = 2147483647
        } else if res < -2147483648 {
            res = -2147483648
        }
        return res
    }
    
    /**
     328. 奇偶链表
     
     给定一个单链表，把所有的奇数节点和偶数节点分别排在一起。请注意，这里的奇数节点和偶数节点指的是节点编号的奇偶性，而不是节点的值的奇偶性。

     请尝试使用原地算法完成。你的算法的空间复杂度应为 O(1)，时间复杂度应为 O(nodes)，nodes 为节点总数。

     示例 1:

     输入: 1->2->3->4->5->NULL
     输出: 1->3->5->2->4->NULL
     示例 2:

     输入: 2->1->3->5->6->4->7->NULL
     输出: 2->3->6->7->1->5->4->NULL
     说明:

     应当保持奇数节点和偶数节点的相对顺序。
     链表的第一个节点视为奇数节点，第二个节点视为偶数节点，以此类推。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/odd-even-linked-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func oddEvenList(_ head: ListNode?) -> ListNode? {
        if nil == head || nil == head?.next || nil == head?.next?.next {
            return head
        }
        
        let dummy = ListNode(0)
        dummy.next = head
        var node1 = head
        var node2 = head?.next
        let node3 = node2
        while nil != node1?.next && nil != node2?.next {
            node1?.next = node2?.next
            node1 = node2?.next
            node2?.next = node1?.next
            node2 = node1?.next
        }
        node1?.next = node3
        return dummy.next
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
      

     提示：

     1 <= prices.length <= 3 * 10 ^ 4
     0 <= prices[i] <= 10 ^ 4

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit2(_ prices: [Int]) -> Int {
        guard prices.count > 1 else {
            return 0
        }
        
        // dp[i][0] 表示第 i 天不持有股票的最大利润，dp[i][1] 表示第 i 天持有股票的最大利润
        var dp = [[Int]](repeating: [Int](repeating: 0, count: 2), count: prices.count)
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        for i in 1 ..< prices.count {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        }
        return dp[prices.count - 1][0]
    }
    
    /**
     14. 最长公共前缀
     
     编写一个函数来查找字符串数组中的最长公共前缀。

     如果不存在公共前缀，返回空字符串 ""。

     示例 1:

     输入: ["flower","flow","flight"]
     输出: "fl"
     示例 2:

     输入: ["dog","racecar","car"]
     输出: ""
     解释: 输入不存在公共前缀。
     说明:

     所有输入只包含小写字母 a-z 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/longest-common-prefix
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func longestCommonPrefix(_ strs: [String]) -> String {
//        guard strs.count > 0 else {
//            return ""
//        }
//
//        var len = Int.max
//        for str in strs {
//            len = min(len, str.count)
//        }
//        var prefix = ""
//        for i in 0 ..< len {
//            let char = Array(strs[0])[i]
//            for j in 1 ..< strs.count {
//                if Array(strs[j])[i] != char {
//                    return prefix
//                }
//            }
//            prefix.append(char)
//        }
//        return prefix
        
        guard strs.count > 0 else {
            return ""
        }
        
        var prefix = strs[0]
        for i in 1 ..< strs.count {
            prefix = longestCommonPrefix(Array(prefix), Array(strs[i]))
            if 0 == prefix.count {
                return prefix
            }
        }
        return prefix
    }
    
    private func longestCommonPrefix(_ chars1: [Character], _ chars2: [Character]) -> String {
        var prefix = ""
        for i in 0 ..< min(chars1.count, chars2.count) {
            if i < chars1.count && i < chars2.count && chars1[i] == chars2[i] {
                prefix.append(chars1[i])
            } else {
                break
            }
        }
        return prefix
    }
    
    /**
     344. 反转字符串
     
     编写一个函数，其作用是将输入的字符串反转过来。输入字符串以字符数组 char[] 的形式给出。

     不要给另外的数组分配额外的空间，你必须原地修改输入数组、使用 O(1) 的额外空间解决这一问题。

     你可以假设数组中的所有字符都是 ASCII 码表中的可打印字符。

      

     示例 1：

     输入：["h","e","l","l","o"]
     输出：["o","l","l","e","h"]
     示例 2：

     输入：["H","a","n","n","a","h"]
     输出：["h","a","n","n","a","H"]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-string
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reverseString(_ s: inout [Character]) {
        guard s.count > 1 else {
            return
        }
        
        let n = s.count
        for i in 0 ..< n/2 {
            let temp = s[i]
            s[i] = s[n - 1 - i]
            s[n - 1 - i] = temp
        }
    }
    
    /**
     541. 反转字符串 II
     
     给定一个字符串 s 和一个整数 k，你需要对从字符串开头算起的每隔 2k 个字符的前 k 个字符进行反转。

     如果剩余字符少于 k 个，则将剩余字符全部反转。
     如果剩余字符小于 2k 但大于或等于 k 个，则反转前 k 个字符，其余字符保持原样。
      

     示例:

     输入: s = "abcdefg", k = 2
     输出: "bacdfeg"
      

     提示：

     该字符串只包含小写英文字母。
     给定字符串的长度和 k 在 [1, 10000] 范围内。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-string-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reverseStr(_ s: String, _ k: Int) -> String {
        guard s.count > 0 && k > 0 else {
            return s
        }
        
        let sArr = Array(s)
        let n = sArr.count
        var resArr = [Character]()
        var index = 0
        while index < n {
            if index + k >= n {
                var arr = Array(sArr[index ... n - 1])
                reverseString(&arr)
                resArr += arr
            } else if index + 2 * k >= n {
                var arr = Array(sArr[index ..< index + k])
                reverseString(&arr)
                resArr += arr
                resArr += Array(sArr[index + k ... n - 1])
            } else {
                var arr = Array(sArr[index ..< index + k])
                reverseString(&arr)
                resArr += arr
                resArr += Array(sArr[index + k ..< index + 2 * k])
            }
            index += 2 * k
        }
        
        var res = ""
        for char in resArr {
            res.append(char)
        }
        return res
    }
    
    /**
     151. 翻转字符串里的单词
     
     给定一个字符串，逐个翻转字符串中的每个单词。

     说明：

     无空格字符构成一个 单词 。
     输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
     如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
      

     示例 1：

     输入："the sky is blue"
     输出："blue is sky the"
     示例 2：

     输入："  hello world!  "
     输出："world! hello"
     解释：输入字符串可以在前面或者后面包含多余的空格，但是反转后的字符不能包括。
     示例 3：

     输入："a good   example"
     输出："example good a"
     解释：如果两个单词间有多余的空格，将反转后单词间的空格减少到只含一个。
     示例 4：

     输入：s = "  Bob    Loves  Alice   "
     输出："Alice Loves Bob"
     示例 5：

     输入：s = "Alice does not even like bob"
     输出："bob like even not does Alice"
      

     提示：

     1 <= s.length <= 104
     s 包含英文大小写字母、数字和空格 ' '
     s 中 至少存在一个 单词
      

     进阶：

     请尝试使用 O(1) 额外空间复杂度的原地解法。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-words-in-a-string
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reverseWords(_ s: String) -> String {
        guard s.count > 0 else {
            return s
        }
        
        var sArr = Array(s)
        reverseString(&sArr)
        var resArr = [Character]()
        var words = [Character]()
        for char in sArr {
            if " " == char {
                if words.count > 0 {
                    reverseString(&words)
                    resArr += words
                    words.removeAll()
                }
                
                if 0 == resArr.count || " " == resArr[resArr.count - 1] {
                    continue
                }
                
                resArr.append(char)
            } else {
                words.append(char)
            }
        }
        
        if words.count > 0 {
            reverseString(&words)
            resArr += words
        }
        
        while resArr.count > 0 && " " == resArr[resArr.count - 1] {
            resArr.removeLast()
        }
        
        var res = ""
        for char in resArr {
            res.append(char)
        }
        return res
    }
    
    func reverseWords2(_ s: String) -> String {
//        guard s.count > 0 else {
//            return s
//        }
//
//        var res = ""
//        var words = [Character]()
//        for char in Array(s) {
//            if " " == char {
//                reverseString(&words)
//                for word in words {
//                    res.append(word)
//                }
//                words.removeAll()
//                res.append(char)
//            } else {
//                words.append(char)
//            }
//        }
//
//        reverseString(&words)
//        for word in words {
//            res.append(word)
//        }
//
//        return res
        
        guard s.count > 0 else {
            return s
        }
        
        let arr = s.components(separatedBy: " ")
        var resArr = [Character]()
        for i in 0 ..< arr.count {
            var tempS = Array(arr[i])
            reverseString(&tempS)
            resArr += tempS
            if i < arr.count - 1 {
                resArr.append(" ")
            }
        }
        
        var res = ""
        for char in resArr {
            res.append(char)
        }
        return res
    }
    
    /**
     917. 仅仅反转字母
     
     给定一个字符串 S，返回 “反转后的” 字符串，其中不是字母的字符都保留在原地，而所有字母的位置发生反转。

      

     示例 1：

     输入："ab-cd"
     输出："dc-ba"
     示例 2：

     输入："a-bC-dEf-ghIj"
     输出："j-Ih-gfE-dCba"
     示例 3：

     输入："Test1ng-Leet=code-Q!"
     输出："Qedo1ct-eeLg=ntse-T!"
      

     提示：

     S.length <= 100
     33 <= S[i].ASCIIcode <= 122
     S 中不包含 \ or "

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-only-letters
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reverseOnlyLetters(_ S: String) -> String {
        guard S.count > 1 else {
            return S
        }
        
        var resArr = Array(S)
        var i = 0, j = resArr.count - 1
        while i < j {
            if !isLetter(resArr[i]) {
                i += 1
                continue
            }
            
            if !isLetter(resArr[j]) {
                j -= 1
                continue
            }
            
            let temp = resArr[i]
            resArr[i] = resArr[j]
            resArr[j] = temp
            i += 1
            j -= 1
        }
        
        var res = ""
        for char in resArr {
            res.append(char)
        }
        return res
    }
    
    private func isLetter(_ c: Character) -> Bool {
        if let asciiValue = c.asciiValue {
            if (asciiValue >= 65 && asciiValue <= 90) || (asciiValue >= 97 && asciiValue <= 122) {
                return true
            }
        }
        
        return false
    }
    
    /**
     1122. 数组的相对排序
     
     给你两个数组，arr1 和 arr2，

     arr2 中的元素各不相同
     arr2 中的每个元素都出现在 arr1 中
     对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。

      

     示例：

     输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
     输出：[2,2,2,1,4,3,3,9,6,7,19]
      

     提示：

     arr1.length, arr2.length <= 1000
     0 <= arr1[i], arr2[i] <= 1000
     arr2 中的元素 arr2[i] 各不相同
     arr2 中的每个元素 arr2[i] 都出现在 arr1 中

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/relative-sort-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func relativeSortArray(_ arr1: [Int], _ arr2: [Int]) -> [Int] {
        // 由于数据大小限制在 0 - 1000，故使用桶排序
        guard arr2.count > 0 else {
            return arr1.sorted(by: <)
        }
        
        var arr = [Int](repeating: 0, count: 1001)
        for num in arr1 {
            arr[num] += 1
        }
        
        var res = [Int]()
        for num in arr2 {
            var count = arr[num]
            while count > 0 {
                res.append(num)
                count -= 1
            }
            arr[num] = 0
        }
        
        if res.count < arr1.count {
            for i in 0 ..< arr.count {
                var count = arr[i]
                while count > 0 {
                    res.append(i)
                    count -= 1
                }
            }
        }
        
        return res
    }
    
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
//        var dict = [Character: Int]()
//        for char in Array(s) {
//            if let count = dict[char] {
//                dict[char] = count + 1
//            } else {
//                dict[char] = 1
//            }
//        }
//
//        for char in Array(t) {
//            if let count = dict[char] {
//                dict[char] = count - 1
//                if 1 == count {
//                    dict.removeValue(forKey: char)
//                }
//            } else {
//                return false
//            }
//        }
//
//        return 0 == dict.count
        
        guard s.count == t.count else {
            return false
        }
        
        var count = [Int](repeating: 0, count: 26)
        let sArr = Array(s)
        let tArr = Array(t)
        for i in 0 ..< sArr.count {
            let sC = Int(sArr[i].asciiValue!)
            let tC = Int(tArr[i].asciiValue!)
            count[sC - 97] += 1
            count[tC - 97] -= 1
        }
        
        for i in 0 ..< 26 {
            if 0 != count[i] {
                return false
            }
        }
        
        return true
    }
    
    /**
     238. 除自身以外数组的乘积
     
     给你一个长度为 n 的整数数组 nums，其中 n > 1，返回输出数组 output ，其中 output[i] 等于 nums 中除 nums[i] 之外其余各元素的乘积。

      

     示例:

     输入: [1,2,3,4]
     输出: [24,12,8,6]
      

     提示：题目数据保证数组之中任意元素的全部前缀元素和后缀（甚至是整个数组）的乘积都在 32 位整数范围内。

     说明: 请不要使用除法，且在 O(n) 时间复杂度内完成此题。

     进阶：
     你可以在常数空间复杂度内完成这个题目吗？（ 出于对空间复杂度分析的目的，输出数组不被视为额外空间。）

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/product-of-array-except-self
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func productExceptSelf(_ nums: [Int]) -> [Int] {
        let n = nums.count
        guard n > 1 else {
            return nums
        }
        
        var res = [Int](repeating: 1, count: n)
        var p = 1, q = 1
        for i in 0 ..< n {
            res[i] = p
            p *= nums[i]
        }
        
        for i in (0 ..< n).reversed() {
            res[i] *= q
            q *= nums[i]
        }
        
        return res
    }
    
    /**
     438. 找到字符串中所有字母异位词
     
     给定一个字符串 s 和一个非空字符串 p，找到 s 中所有是 p 的字母异位词的子串，返回这些子串的起始索引。

     字符串只包含小写英文字母，并且字符串 s 和 p 的长度都不超过 20100。

     说明：

     字母异位词指字母相同，但排列不同的字符串。
     不考虑答案输出的顺序。
     示例 1:

     输入:
     s: "cbaebabacd" p: "abc"

     输出:
     [0, 6]

     解释:
     起始索引等于 0 的子串是 "cba", 它是 "abc" 的字母异位词。
     起始索引等于 6 的子串是 "bac", 它是 "abc" 的字母异位词。
      示例 2:

     输入:
     s: "abab" p: "ab"

     输出:
     [0, 1, 2]

     解释:
     起始索引等于 0 的子串是 "ab", 它是 "ab" 的字母异位词。
     起始索引等于 1 的子串是 "ba", 它是 "ab" 的字母异位词。
     起始索引等于 2 的子串是 "ab", 它是 "ab" 的字母异位词。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/find-all-anagrams-in-a-string
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findAnagrams(_ s: String, _ p: String) -> [Int] {
        guard p.count > 0 && s.count >= p.count else {
            return [Int]()
        }
        
        // 滑动窗口解法
        let sArr = Array(s), pArr = Array(p), pLen = pArr.count
        var sCount = [Int](repeating: 0, count: 26)
        var pCount = [Int](repeating: 0, count: 26)
        for i in 0 ..< pLen {
            let sIndex = Int(sArr[i].asciiValue!)
            sCount[sIndex - 97] += 1
            let pIndex = Int(pArr[i].asciiValue!)
            pCount[pIndex - 97] += 1
        }
        
        var res = [Int]()
        if isAnagram(sCount, pCount) {
            res.append(0)
        }
        
        if sArr.count - pLen < 1 {
            return res
        }
        
        for i in 1 ... sArr.count - pLen {
            let firstIndex = Int(sArr[i - 1].asciiValue!)
            sCount[firstIndex - 97] -= 1
            let lastIndex = Int(sArr[i + pLen - 1].asciiValue!)
            sCount[lastIndex - 97] += 1
            if isAnagram(sCount, pCount) {
                res.append(i)
            }
        }
        
        return res
    }
    
    private func isAnagram(_ sArr: [Int], _ pArr: [Int]) -> Bool {
        guard sArr.count == pArr.count else {
            return false
        }
        
        for i in 0 ..< sArr.count {
            if sArr[i] != pArr[i] {
                return false
            }
        }
        
        return true
    }
    
    /**
     125. 验证回文串
     
     给定一个字符串，验证它是否是回文串，只考虑字母和数字字符，可以忽略字母的大小写。

     说明：本题中，我们将空字符串定义为有效的回文串。

     示例 1:

     输入: "A man, a plan, a canal: Panama"
     输出: true
     示例 2:

     输入: "race a car"
     输出: false

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-palindrome
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isPalindrome(_ s: String) -> Bool {
        guard s.count > 1 else {
            return true
        }
        
        let arr = Array(s)
        var i = 0, j = arr.count - 1
        while i < j {
            if !isLetterOrDigit(arr[i]) {
                i += 1
                continue
            }
            
            if !isLetterOrDigit(arr[j]) {
                j -= 1
                continue
            }
            
            let pre = arr[i].asciiValue!
            let next = arr[j].asciiValue!
            // pre|32、next|32 表示将字母转换为小写字母
            if (pre == next) || ((pre|32) == (next|32)) {
                i += 1
                j -= 1
                continue
            }
            
            return false
        }
        return true
    }
    
    private func isLetterOrDigit(_ c: Character) -> Bool {
        if let asciiValue = c.asciiValue {
            return
                (asciiValue >= 65 && asciiValue <= 90) ||
                (asciiValue >= 97 && asciiValue <= 122) ||
                (asciiValue >= 48 && asciiValue <= 57)
        }
        
        return false
    }
    
    /**
     402. 移掉K位数字
     
     给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。

     注意:

     num 的长度小于 10002 且 ≥ k。
     num 不会包含任何前导零。
     示例 1 :

     输入: num = "1432219", k = 3
     输出: "1219"
     解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
     示例 2 :

     输入: num = "10200", k = 1
     输出: "200"
     解释: 移掉首位的 1 剩下的数字为 200. 注意输出不能有任何前导零。
     示例 3 :

     输入: num = "10", k = 2
     输出: "0"
     解释: 从原数字移除所有的数字，剩余为空就是0。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/remove-k-digits
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func removeKdigits(_ num: String, _ k: Int) -> String {
//        guard num.count > 0 && num.count > k else {
//            return "0"
//        }
//
//        var nums = Array(num)
//        var len = k
//        while len > 0 {
//            var index = -1
//            for i in 0 ..< nums.count - 1 {
//                if nums[i] > nums[i + 1] {
//                    index = i
//                    break
//                }
//            }
//            if -1 == index {
//                index = nums.count - 1
//            }
//            nums.remove(at: index)
//            len -= 1
//        }
//
//        var res = ""
//        for char in nums {
//            if "0" == char && 0 == res.count {
//                continue
//            }
//            res.append(char)
//        }
//        return "" == res ? "0" : res
        
        // 单调栈解法，单调不降
        guard k > 0 else {
            return num
        }
        
        let n = num.count
        guard n > k else {
            return "0"
        }
        
        let nums = Array(num)
        var stack = [Character]()
        var len = k
        for i in 0 ..< n {
            let c = nums[i]
            while !stack.isEmpty && len > 0 && stack.last! > c {
                stack.removeLast()
                len -= 1
            }
            stack.append(c)
        }
        
        while len > 0 {
            stack.removeLast()
            len -= 1
        }
        
        var res = ""
        while !stack.isEmpty {
            let char = stack.removeFirst()
            if "0" == char && 0 == res.count {
                continue
            }
            res.append(char)
        }
        return "" == res ? "0" : res
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

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sliding-window-maximum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        if k < 2 {
            return nums
        }
        
        if k >= nums.count {
            var max = Int.min
            for num in nums {
                if num > max {
                    max = num
                }
            }
            return [max]
        }
        
        // 双端队列
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
            // 保证窗口大小 <= k
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
     680. 验证回文字符串 Ⅱ
     
     给定一个非空字符串 s，最多删除一个字符。判断是否能成为回文字符串。

     示例 1:

     输入: "aba"
     输出: True
     示例 2:

     输入: "abca"
     输出: True
     解释: 你可以删除c字符。
     注意:

     字符串只包含从 a-z 的小写字母。字符串的最大长度是50000。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-palindrome-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func validPalindrome(_ s: String) -> Bool {
        let sArr = Array(s)
        let n = sArr.count
        guard n > 1 else {
            return true
        }
        
        var low = 0, high = n - 1
        while low < high {
            if sArr[low] == sArr[high] {
                low += 1
                high -= 1
            } else {
                var flag1 = true, flag2 = true
                // 删掉右边字符
                var i = low, j = high - 1
                while i < j {
                    if sArr[i] == sArr[j] {
                        i += 1
                        j -= 1
                    } else {
                        flag1 = false
                        break
                    }
                }
                
                // 删掉左边字符
                i = low + 1
                j = high
                while i < j {
                    if sArr[i] == sArr[j] {
                        i += 1
                        j -= 1
                    } else {
                        flag2 = false
                        break
                    }
                }
                
                return flag1 || flag2
            }
        }
        
        return true
    }
    
    /**
     5. 最长回文子串
     
     给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

     示例 1：

     输入: "babad"
     输出: "bab"
     注意: "aba" 也是一个有效答案。
     示例 2：

     输入: "cbbd"
     输出: "bb"

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/longest-palindromic-substring
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func longestPalindrome(_ s: String) -> String {
        guard s.count > 1 else {
            return s
        }
        
        let sArr = Array(s)
        let n = s.count
        var maxLen = 0
        var low = 0, high = 0
        // dp[i][j] 表示 s[i ... j] 是否为回文串
        var dp = [[Bool]](repeating: [Bool](repeating: false, count: n), count: n)
        for i in (0 ..< n).reversed() {
            for j in i ..< n {
                dp[i][j] = sArr[i] == sArr[j] && (j - i < 2 || dp[i + 1][j - 1])
                if dp[i][j] && j - i + 1 > maxLen {
                    maxLen = j - i + 1
                    low = i
                    high = j
                }
            }
        }
        return String(s[s.index(s.startIndex, offsetBy: low) ... s.index(s.startIndex, offsetBy: high)])
    }
    
    /**
     205. 同构字符串
     
     给定两个字符串 s 和 t，判断它们是否是同构的。

     如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。

     所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

     示例 1:

     输入: s = "egg", t = "add"
     输出: true
     示例 2:

     输入: s = "foo", t = "bar"
     输出: false
     示例 3:

     输入: s = "paper", t = "title"
     输出: true
     说明:
     你可以假设 s 和 t 具有相同的长度。



     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/isomorphic-strings
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isIsomorphic(_ s: String, _ t: String) -> Bool {
        guard s.count == t.count else {
            return false
        }
        
        let sArr = Array(s)
        let tArr = Array(t)
        var dict1 = [Character: Character]()
        var dict2 = [Character: Character]()
        for i in 0 ..< sArr.count {
            let sc = sArr[i]
            let tc = tArr[i]
            if let c = dict1[sc] {
                if c != tc {
                    return false
                }
            } else {
                dict1[sc] = tc
            }
            
            if let c = dict2[tc] {
                if c != sc {
                    return false
                }
            } else {
                dict2[tc] = sc
            }
        }
        
        return true
    }
    
    /**
     44. 通配符匹配
     
     给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配。

     '?' 可以匹配任何单个字符。
     '*' 可以匹配任意字符串（包括空字符串）。
     两个字符串完全匹配才算匹配成功。

     说明:

     s 可能为空，且只包含从 a-z 的小写字母。
     p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 *。
     示例 1:

     输入:
     s = "aa"
     p = "a"
     输出: false
     解释: "a" 无法匹配 "aa" 整个字符串。
     示例 2:

     输入:
     s = "aa"
     p = "*"
     输出: true
     解释: '*' 可以匹配任意字符串。
     示例 3:

     输入:
     s = "cb"
     p = "?a"
     输出: false
     解释: '?' 可以匹配 'c', 但第二个 'a' 无法匹配 'b'。
     示例 4:

     输入:
     s = "adceb"
     p = "*a*b"
     输出: true
     解释: 第一个 '*' 可以匹配空字符串, 第二个 '*' 可以匹配字符串 "dce".
     示例 5:

     输入:
     s = "acdcb"
     p = "a*c?b"
     输出: false

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/wildcard-matching
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isMatch(_ s: String, _ p: String) -> Bool {
        let sArr = Array(s), pArr = Array(p)
        let m = sArr.count, n = pArr.count
        if n < 1 {
            return m < 1
        }
        
        // dp[i][j] 表示字符串 s 的前 i 个字符和模式 p 的前 j 个字符是否能匹配
        var dp = [[Bool]](repeating: [Bool](repeating: false, count: n + 1), count: m + 1)
        dp[0][0] = true
        for i in 1 ... n {
            if pArr[i - 1] == "*" {
                dp[0][i] = true
            } else {
                break
            }
        }
        
        if m < 1 {
            return dp[m][n]
        }
        
        for i in 1 ... m {
            for j in 1 ... n {
                if pArr[j - 1] == "*" {
                    dp[i][j] = dp[i - 1][j] || dp[i][j - 1]
                } else if pArr[j - 1] == "?" || sArr[i - 1] == pArr[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1]
                }
            }
        }
        return dp[m][n]
    }
    
    /**
     10. 正则表达式匹配
     
     给你一个字符串 s 和一个字符规律 p，请你来实现一个支持 '.' 和 '*' 的正则表达式匹配。

     '.' 匹配任意单个字符
     '*' 匹配零个或多个前面的那一个元素
     所谓匹配，是要涵盖 整个 字符串 s的，而不是部分字符串。

      
     示例 1：

     输入：s = "aa" p = "a"
     输出：false
     解释："a" 无法匹配 "aa" 整个字符串。
     示例 2:

     输入：s = "aa" p = "a*"
     输出：true
     解释：因为 '*' 代表可以匹配零个或多个前面的那一个元素, 在这里前面的元素就是 'a'。因此，字符串 "aa" 可被视为 'a' 重复了一次。
     示例 3：

     输入：s = "ab" p = ".*"
     输出：true
     解释：".*" 表示可匹配零个或多个（'*'）任意字符（'.'）。
     示例 4：

     输入：s = "aab" p = "c*a*b"
     输出：true
     解释：因为 '*' 表示零个或多个，这里 'c' 为 0 个, 'a' 被重复一次。因此可以匹配字符串 "aab"。
     示例 5：

     输入：s = "mississippi" p = "mis*is*p*."
     输出：false
      

     提示：

     0 <= s.length <= 20
     0 <= p.length <= 30
     s 可能为空，且只包含从 a-z 的小写字母。
     p 可能为空，且只包含从 a-z 的小写字母，以及字符 . 和 *。
     保证每次出现字符 * 时，前面都匹配到有效的字符


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/regular-expression-matching
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isMatch2(_ s: String, _ p: String) -> Bool {
        let sArr = Array(s), pArr = Array(p)
        let m = sArr.count, n = pArr.count
        if n < 1 {
            return m < 1
        }
        
        // dp[i][j] 表示字符串 s 的前 i 个字符和模式 p 的前 j 个字符是否能匹配
        var dp = [[Bool]](repeating: [Bool](repeating: false, count: n + 1), count: m + 1)
        dp[0][0] = true
        for i in 0 ... m {
            for j in 1 ... n {
                if "*" == pArr[j - 1] {
                    dp[i][j] = dp[i][j - 2]
                    if matches(sArr, pArr, i, j - 1) {
                        dp[i][j] = dp[i][j] || dp[i - 1][j]
                    }
                } else {
                    if matches(sArr, pArr, i, j) {
                        dp[i][j] = dp[i - 1][j - 1]
                    }
                }
            }
        }
        return dp[m][n]
    }
    
    private func matches(_ sArr: [Character], _ pArr: [Character], _ i: Int, _ j: Int) -> Bool {
        if 0 == i {
            return false
        }
        
        if "." == pArr[j - 1] {
            return true
        }
        
        return sArr[i - 1] == pArr[j - 1]
    }
}
