//
//  Homework_week_06.swift
//  DataStructDemo
//
//  Created by 刘练 on 2020/10/8.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

class TrieNode {
    var children: [Character : TrieNode] = [:]
    var word: String?
}

class Homework_week_06 {
    /**
     动态规划要点：
     a、找重复性
     b、定义状态数组 - dp[i][j] 的含义一定要清晰且正确
     c、推导 DP 方程
     */
    
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
            dp.append([Int](repeating: 0, count: n))
        }
                
        for i in 0 ..< m {
            dp[i][n - 1] = 1
        }
        
        for j in 0 ..< n {
            dp[m - 1][j] = 1
        }
        
        for i in (0 ..< m - 1).reversed() {
            for j in (0 ..< n - 1).reversed() {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1]
            }
        }
        
        return dp[0][0]
    }
    
    /**
     63. 不同路径 II
     
     一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

     机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

     现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？



     网格中的障碍物和空位置分别用 1 和 0 来表示。

     说明：m 和 n 的值均不超过 100。

     示例 1:

     输入:
     [
       [0,0,0],
       [0,1,0],
       [0,0,0]
     ]
     输出: 2
     解释:
     3x3 网格的正中间有一个障碍物。
     从左上角到右下角一共有 2 条不同的路径：
     1. 向右 -> 向右 -> 向下 -> 向下
     2. 向下 -> 向下 -> 向右 -> 向右

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/unique-paths-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func uniquePathsWithObstacles(_ obstacleGrid: [[Int]]) -> Int {
        let m = obstacleGrid.count
        guard m > 0 else {
            return 0
        }
        
        let n = obstacleGrid[0].count
        guard n > 0 else {
            return 0
        }
        
        guard 1 != obstacleGrid[m - 1][n - 1] else {
            return 0
        }
        
        var dp = [[Int]]()
        for _ in 0 ..< m {
            dp.append([Int](repeating: 0, count: n))
        }
        
        dp[m - 1][n - 1] = 1
        
        for i in (0 ..< m - 1).reversed() {
            if 1 == obstacleGrid[i][n - 1] {
                break
            }
            dp[i][n - 1] = 1
        }
        
        for j in (0 ..< n - 1).reversed() {
            if 1 == obstacleGrid[m - 1][j] {
                break
            }
            dp[m - 1][j] = 1
        }
        
        for i in (0 ..< m - 1).reversed() {
            for j in (0 ..< n - 1).reversed() {
                if 1 == obstacleGrid[i][j] {
                    dp[i][j] = 0
                } else {
                    dp[i][j] = dp[i + 1][j] + dp[i][j + 1]
                }
            }
        }
        
        return dp[0][0]
    }
    
    /**
     980. 不同路径 III
     
     在二维网格 grid 上，有 4 种类型的方格：

     1 表示起始方格。且只有一个起始方格。
     2 表示结束方格，且只有一个结束方格。
     0 表示我们可以走过的空方格。
     -1 表示我们无法跨越的障碍。
     返回在四个方向（上、下、左、右）上行走时，从起始方格到结束方格的不同路径的数目。

     每一个无障碍方格都要通过一次，但是一条路径中不能重复通过同一个方格。

      

     示例 1：

     输入：[[1,0,0,0],[0,0,0,0],[0,0,2,-1]]
     输出：2
     解释：我们有以下两条路径：
     1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2)
     2. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2)
     示例 2：

     输入：[[1,0,0,0],[0,0,0,0],[0,0,0,2]]
     输出：4
     解释：我们有以下四条路径：
     1. (0,0),(0,1),(0,2),(0,3),(1,3),(1,2),(1,1),(1,0),(2,0),(2,1),(2,2),(2,3)
     2. (0,0),(0,1),(1,1),(1,0),(2,0),(2,1),(2,2),(1,2),(0,2),(0,3),(1,3),(2,3)
     3. (0,0),(1,0),(2,0),(2,1),(2,2),(1,2),(1,1),(0,1),(0,2),(0,3),(1,3),(2,3)
     4. (0,0),(1,0),(2,0),(2,1),(1,1),(0,1),(0,2),(0,3),(1,3),(1,2),(2,2),(2,3)
     示例 3：

     输入：[[0,1],[2,0]]
     输出：0
     解释：
     没有一条路能完全穿过每一个空的方格一次。
     请注意，起始和结束方格可以位于网格中的任意位置。
      

     提示：

     1 <= grid.length * grid[0].length <= 20

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/unique-paths-iii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func uniquePathsIII(_ grid: [[Int]]) -> Int {
        // DFS + 回溯，参考链接：https://leetcode-cn.com/problems/unique-paths-iii/solution/dfs-hui-su-shuang-bai-by-quantum-10/
        let m = grid.count
        guard m > 0 else {
            return 0
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return 0
        }
        
        // 找到起始位置
        var startRow = -1, startCol = -1, steps = 1
        for i in 0 ..< m {
            for j in 0 ..< n {
                if 1 == grid[i][j] {
                    startRow = i
                    startCol = j
                    continue
                }
                
                if 0 == grid[i][j] {
                    steps += 1
                }
            }
        }
        
        var tempGrid = grid
        return dfsUniquePathsIII(&tempGrid, m, n, startRow, startCol, steps)
    }
    
    private func dfsUniquePathsIII(_ grid: inout [[Int]], _ m: Int, _ n: Int, _ row: Int, _ col: Int, _ steps: Int) -> Int {
        if row < 0 || row >= m || col < 0 || col >= n || -1 == grid[row][col] {
            return 0
        }
        
        if 2 == grid[row][col] {
            return 0 == steps ? 1 : 0
        }
        
        grid[row][col] = -1
        var res = 0
        res += dfsUniquePathsIII(&grid, m, n, row - 1, col, steps - 1)
        res += dfsUniquePathsIII(&grid, m, n, row + 1, col, steps - 1)
        res += dfsUniquePathsIII(&grid, m, n, row, col - 1, steps - 1)
        res += dfsUniquePathsIII(&grid, m, n, row, col + 1, steps - 1)
        grid[row][col] = 0  // 回溯
        return res
    }
    
    /**
     1143. 最长公共子序列
     
     给定两个字符串 text1 和 text2，返回这两个字符串的最长公共子序列的长度。

     一个字符串的 子序列 是指这样一个新的字符串：它是由原字符串在不改变字符的相对顺序的情况下删除某些字符（也可以不删除任何字符）后组成的新字符串。
     例如，"ace" 是 "abcde" 的子序列，但 "aec" 不是 "abcde" 的子序列。两个字符串的「公共子序列」是这两个字符串所共同拥有的子序列。

     若这两个字符串没有公共子序列，则返回 0。

      

     示例 1:

     输入：text1 = "abcde", text2 = "ace"
     输出：3
     解释：最长公共子序列是 "ace"，它的长度为 3。
     示例 2:

     输入：text1 = "abc", text2 = "abc"
     输出：3
     解释：最长公共子序列是 "abc"，它的长度为 3。
     示例 3:

     输入：text1 = "abc", text2 = "def"
     输出：0
     解释：两个字符串没有公共子序列，返回 0。
      

     提示:

     1 <= text1.length <= 1000
     1 <= text2.length <= 1000
     输入的字符串只含有小写英文字符。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/longest-common-subsequence
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func longestCommonSubsequence(_ text1: String, _ text2: String) -> Int {
        let text1Arr = Array(text1)
        let text2Arr = Array(text2)
        let m = text1Arr.count
        let n = text2Arr.count
        guard m > 0 && n > 0 else {
            return 0
        }
        
        var dp = [[Int]]()
        for _ in 0 ..< m {
            dp.append([Int](repeating: 0, count: n))
        }
        
        for i in 0 ..< m {
            for j in 0 ..< n {
                let left = i > 0 ? dp[i - 1][j] : 0
                let top = j > 0 ? dp[i][j - 1] : 0
                let leftTop = ((i > 0 && j > 0) ? dp[i - 1][j - 1] : 0)
                if text1Arr[i] == text2Arr[j] {
                    dp[i][j] = leftTop + 1
                } else {
                    dp[i][j] = max(left, top)
                }
            }
        }
        return dp[m - 1][n - 1]
    }
    
    /**
     70. 爬楼梯
     
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
     */
    func climbStairs(_ n: Int) -> Int {
        if n <= 2 {
            return n
        }
        
        var f1 = 1
        var f2 = 2
        var ans = 0
        for _ in 3 ... n {
            ans = f1 + f2
            f1 = f2
            f2 = ans
        }
        return ans
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

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/triangle
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        let m = triangle.count
        guard m > 0 else {
            return 0
        }

        var dp = triangle[m - 1]
        for i in (0 ..< m - 1).reversed() {
            let nums = triangle[i]
            for j in 0 ..< nums.count {
                dp[j] = min(dp[j] + nums[j], dp[j + 1] + nums[j])
            }
        }

        return dp[0]
        
//        let n = triangle.count
//        guard n > 0 else {
//            return Int.min
//        }
//
//        guard n > 1 else {
//            return triangle[0][0]
//        }
//
//        // dp[i][j] 表示点 (i, j) 到最底层的最小路径和
//        var dp = triangle
//        for i in (0 ... n - 2).reversed() {
//            for j in 0 ... i {
//                dp[i][j] += min(dp[i + 1][j], dp[i + 1][j + 1])
//            }
//        }
//        return dp[0][0]
        
//        var memo = [[Int]]()
//        for i in 0 ..< triangle.count {
//            var arr = [Int]()
//            for _ in 0 ..< triangle[i].count {
//                arr.append(Int.min)
//            }
//            memo.append(arr)
//        }
//        return minimumTotalHelper(0, 0, triangle.count, triangle, &memo)
    }
    
    private func minimumTotalHelper(_ i: Int, _ j: Int, _ n: Int, _ triangle: [[Int]], _ memo: inout [[Int]]) -> Int {
        if Int.min != memo[i][j] {
            return memo[i][j]
        }
        
        if i == n - 1 {
            memo[i][j] = triangle[i][j]
            return memo[i][j]
        }
        
        let left = minimumTotalHelper(i + 1, j, n, triangle, &memo)
        let right = minimumTotalHelper(i + 1, j + 1, n, triangle, &memo)
        memo[i][j] = min(left, right) + triangle[i][j]
        return memo[i][j]
    }
    
    /**
     152. 乘积最大子数组
     
     给你一个整数数组 nums ，请你找出数组中乘积最大的连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

      

     示例 1:

     输入: [2,3,-2,4]
     输出: 6
     解释: 子数组 [2,3] 有最大乘积 6。
     示例 2:

     输入: [-2,0,-1]
     输出: 0
     解释: 结果不能为 2, 因为 [-2,-1] 不是子数组。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximum-product-subarray
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProduct(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 0 else {
            return 0
        }
        
        var dpMax = [Int](repeating: 0, count: n)
        var dpMin = [Int](repeating: 0, count: n)
        dpMax[0] = nums[0]
        dpMin[0] = nums[0]
        for i in 1 ..< n {
            dpMax[i] = max(nums[i], nums[i] * dpMax[i - 1], nums[i] * dpMin[i - 1])
            dpMin[i] = min(nums[i], nums[i] * dpMax[i - 1], nums[i] * dpMin[i - 1])
        }
        
        var res = dpMax[0]
        for num in dpMax {
            res = max(res, num)
        }
        return res
    }
    
    /**
     322. 零钱兑换
     
     给定不同面额的硬币 coins 和一个总金额 amount。编写一个函数来计算可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回 -1。

     你可以认为每种硬币的数量是无限的。

      

     示例 1：

     输入：coins = [1, 2, 5], amount = 11
     输出：3
     解释：11 = 5 + 5 + 1
     示例 2：

     输入：coins = [2], amount = 3
     输出：-1
     示例 3：

     输入：coins = [1], amount = 0
     输出：0
     示例 4：

     输入：coins = [1], amount = 1
     输出：1
     示例 5：

     输入：coins = [1], amount = 2
     输出：2
      

     提示：

     1 <= coins.length <= 12
     1 <= coins[i] <= 231 - 1
     0 <= amount <= 231 - 1

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/coin-change
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func coinChange(_ coins: [Int], _ amount: Int) -> Int {
        guard amount > 0 else {
            return 0
        }
        
        guard coins.count > 0 else {
            return -1
        }
        
        /**
        贪心 + dfs 解法
        
        贪心
        11. 想要总硬币数最少，肯定是优先用大面值硬币，所以对 coins 按从大到小排序
        12. 先丢大硬币，再丢会超过总额时，就可以递归下一层丢的是稍小面值的硬币

        乘法对加法的加速
        21. 优先丢大硬币进去尝试，也没必要一个一个丢，可以用乘法算一下最多能丢几个

        k = amount / coins[c_index] 计算最大能投几个
        amount - k * coins[c_index] 减去扔了 k 个硬币
        count + k 加 k 个硬币

        如果因为丢多了导致最后无法凑出总额，再回溯减少大硬币数量
        最先找到的并不是最优解
        31. 注意不是现实中发行的硬币，面值组合规划合理，会有奇葩情况
        32. 考虑到有 [1,7,10] 这种用例，按照贪心思路 10 + 1 + 1 + 1 + 1 会比 7 + 7 更早找到
        33. 所以还是需要把所有情况都递归完

        ans 疯狂剪枝
        41. 贪心虽然得不到最优解，但也不是没用的
        42. 我们快速算出一个贪心的 ans 之后，虽然还会有奇葩情况，但是绝大部分普通情况就可以疯狂剪枝了
        */
        let sortedCoins = coins.sorted(by: >)
        var res = Int.max
        dfsCoinChange(sortedCoins, amount, 0, 0, &res)
        return Int.max == res ? -1 : res
    }
    
    private func dfsCoinChange(_ coins: [Int], _ amount: Int, _ index: Int, _ count: Int, _ res: inout Int) {
        if 0 == amount {
            res = min(res, count)
            return
        }
        
        if index == coins.count {
            return
        }
        
        var k = amount/coins[index]
        while k >= 0 && k + count < res {
            dfsCoinChange(coins, amount - k * coins[index], index + 1, k + count, &res)
            // 使用最大金额面值的零钱无法凑成总金额时，使用次大金额面值进行拼凑
            k -= 1
        }
    }
    
    /**
     使用动态规划法解决 零钱兑换 问题
     */
    func coinChangeWithDP(_ coins: [Int], _ amount: Int) -> Int {
        guard amount > 0 else {
            return 0
        }
        
        guard coins.count > 0 else {
            return -1
        }
        
        // dp[i] 表示凑成金额 i 时所需的最少硬币数
        // 用 amount + 1 填充 dp，amount + 1 表示不可能达到的换取数量
        var dp = [Int](repeating: amount + 1, count: amount + 1)
        dp[0] = 0   // 凑成金额 0 时所需的最少硬币数肯定为 0
        for i in 1 ... amount {
            for j in 0 ..< coins.count {
                if i - coins[j] >= 0 {
                    dp[i] = min(dp[i], dp[i - coins[j]] + 1)
                }
            }
        }
        
        return amount + 1 == dp[amount] ? -1 : dp[amount]
    }
    
    /**
     518. 零钱兑换 II
     
     给定不同面额的硬币和一个总金额。写出函数来计算可以凑成总金额的硬币组合数。假设每一种面额的硬币有无限个。

      

     示例 1:

     输入: amount = 5, coins = [1, 2, 5]
     输出: 4
     解释: 有四种方式可以凑成总金额:
     5=5
     5=2+2+1
     5=2+1+1+1
     5=1+1+1+1+1
     示例 2:

     输入: amount = 3, coins = [2]
     输出: 0
     解释: 只用面额2的硬币不能凑成总金额3。
     示例 3:

     输入: amount = 10, coins = [10]
     输出: 1
      

     注意:

     你可以假设：

     0 <= amount (总金额) <= 5000
     1 <= coin (硬币面额) <= 5000
     硬币种类不超过 500 种
     结果符合 32 位符号整数

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/coin-change-2
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func change(_ amount: Int, _ coins: [Int]) -> Int {
        // 解法参考：https://leetcode-cn.com/problems/coin-change-2/solution/ling-qian-dui-huan-iihe-pa-lou-ti-wen-ti-dao-di-yo/
        if 0 == amount  {
            return 1
        }
        
//        let m = coins.count + 1
//        let n = amount + 1
//        // dp[i][j] 表示前 i 个硬币凑成金额 j 的组合数
//        var dp = [[Int]]()
//        for _ in 0 ..< m {
//            dp.append([Int](repeating: 0, count: n))
//        }
//
//        // 初始化
//        for i in 0 ..< m {
//            dp[i][0] = 1
//        }
//
//        for i in 1 ..< m {
//            for j in 1 ..< n {
//                if j >= coins[i - 1] {
//                    dp[i][j] = dp[i][j - coins[i - 1]] + dp[i - 1][j]
//                } else {
//                    dp[i][j] = dp[i - 1][j]
//                }
//            }
//        }
//
//        return dp[m - 1][n - 1]
        
        let n = amount
        var dp = [Int](repeating: 0, count: n + 1)
        dp[0] = 1
        for coin in coins {
            for i in 1 ... amount {
                if i >= coin {
                    dp[i] += dp[i - coin]
                }
            }
        }
        return dp[n]
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
        guard nums.count > 0 else {
            return 0
        }
        
        if 1 == nums.count {
            return nums[0]
        }
        
        if 2 == nums.count {
            return max(nums[0], nums[1])
        }
        
        let n = nums.count
        // dp[i] 表示偷窃第 i 号房间的最高金额
        var dp = [Int](repeating: 0, count: n)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in 2 ..< n {
            // 偷窃第 i 号房间时，要么偷窃 i 及 i - 2，要么偷窃 i - 1
            dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])
        }
        return dp[n - 1]
    }
    
    /**
     213. 打家劫舍 II
     
     你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都围成一圈，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警。

     给定一个代表每个房屋存放金额的非负整数数组，计算你在不触动警报装置的情况下，能够偷窃到的最高金额。

     示例 1:

     输入: [2,3,2]
     输出: 3
     解释: 你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。
     示例 2:

     输入: [1,2,3,1]
     输出: 4
     解释: 你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。
          偷窃到的最高金额 = 1 + 3 = 4 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/house-robber-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func rob2(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 0 else {
            return 0
        }

        if n < 3 {
            return 1 == n ? nums[0] : max(nums[0], nums[1])
        }

        // 不偷最后一间房
        var dp1 = [Int](repeating: 0, count: n)
        dp1[0] = nums[0]
        dp1[1] = max(nums[0], nums[1])
        // 不偷第一间房
        var dp2 = [Int](repeating: 0, count: n)
        dp2[1] = nums[1]
        for i in 2 ..< n {
            dp1[i] = max(nums[i] + dp1[i - 2], dp1[i - 1])
            dp2[i] = max(nums[i] + dp2[i - 2], dp2[i - 1])
        }
        return max(dp1[n - 2], dp2[n - 1])
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
        // 暴力法
//        var res = 0
//        let n = prices.count
//        for i in 0 ..< n {
//            for j in i + 1 ..< n {
//                if prices[j] > prices[i] {
//                     res = max(res, prices[j] - prices[i])
//                }
//            }
//        }
//        return res

        // 动态规划法
        guard prices.count > 0 else {
            return 0
        }

        var minPrice = prices[0]
        let n = prices.count
        var dp = [Int](repeating: 0, count: n)
        for i in 1 ..< n {
            minPrice = min(minPrice, prices[i])
            if prices[i] > minPrice {
                dp[i] = prices[i] - minPrice
            }
        }

        var res = dp[0]
        for i in 1 ..< n {
            res = max(res, dp[i])
        }
        return res
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
        var res = 0
        for i in 1 ..< prices.count {
            if prices[i] > prices[i - 1] {
                res += (prices[i] - prices[i - 1])
            }
        }
        return res
    }
    
    /**
     123. 买卖股票的最佳时机 III
     
     给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

     设计一个算法来计算你所能获取的最大利润。你最多可以完成 两笔 交易。

     注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

     示例 1:

     输入: [3,3,5,0,0,3,1,4]
     输出: 6
     解释: 在第 4 天（股票价格 = 0）的时候买入，在第 6 天（股票价格 = 3）的时候卖出，这笔交易所能获得利润 = 3-0 = 3 。
          随后，在第 7 天（股票价格 = 1）的时候买入，在第 8 天 （股票价格 = 4）的时候卖出，这笔交易所能获得利润 = 4-1 = 3 。
     示例 2:

     输入: [1,2,3,4,5]
     输出: 4
     解释: 在第 1 天（股票价格 = 1）的时候买入，在第 5 天 （股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5-1 = 4 。
          注意你不能在第 1 天和第 2 天接连购买股票，之后再将它们卖出。
          因为这样属于同时参与了多笔交易，你必须在再次购买前出售掉之前的股票。
     示例 3:

     输入: [7,6,4,3,1]
     输出: 0
     解释: 在这个情况下, 没有交易完成, 所以最大利润为 0。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit3(_ prices: [Int]) -> Int {
        guard prices.count > 0 else {
            return 0
        }
        
        let n = prices.count
        /**
         详情参考题解：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/solution/wu-chong-shi-xian-xiang-xi-tu-jie-123mai-mai-gu-pi/ 中的 动态规划 二维数组
         定义一个二维数组 dp[n][5]，n 表示天数，5 表示 5 种不同的状态：初始 -> 买入 1 -> 卖出 1 -> 买入 2 -> 卖出 2
         dp[i][0] 初始化状态
         dp[i][1] 第一次买入
         dp[i][2] 第一次卖出
         dp[i][3] 第二次买入
         dp[i][4] 第二次卖出
         
         DP 公式推导：
         
         第一次买入：从初始状态转换而来，或者第一次买入后保持不动
         dp[i][1] = max(dp[i-1][1],dp[i-1][0]-prices[i])
                     
         第一次卖出：从第一次买入转换而来，或者第一次卖出后保持不动
         dp[i][2] = max(dp[i-1][2],dp[i-1][1]+prices[i])
         第二次买卖的DP推到如下：


         第二次买入：从第一次卖出转换而来，或者第二次买入后保持不动
         dp[i][3] = max(dp[i-1][3],dp[i-1][2]-prices[i])


         第二次卖出：从第二次买入转换而来，或者第二次卖出后保持不动
         dp[i][4] = max(dp[i-1][4],dp[i-1][3]+prices[i])
         把上面两次买卖推导公式整合到一起就是完整的计算过程了，第一天的初始化过程请查看代码部分。
         最后求的利润最大值就保存在 dp[n-1][0]、dp[n-1][1]、dp[n-1][2]、dp[n-1][3]、dp[n-1][4]中，求出这几个值的max再返回就可以了。
         
         */
        var dp = [[Int]]()
        for _ in 0 ..< n {
            dp.append([Int](repeating: 0, count: 5))
        }
        
        // 初始化第一天的状态
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[0][2] = 0
        dp[0][3] = -prices[0]
        dp[0][4] = 0
        
        for i in 1 ..< n {
            dp[i][0] = dp[i - 1][0]
            // 第一次买
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
            // 第一次卖
            dp[i][2] = max(dp[i - 1][2], dp[i - 1][1] + prices[i])
            // 第二次买
            dp[i][3] = max(dp[i - 1][3], dp[i - 1][2] - prices[i])
            // 第二次卖
            dp[i][4] = max(dp[i - 1][4], dp[i - 1][3] + prices[i])
        }
        
        return max(dp[n - 1][0], dp[n - 1][1], dp[n - 1][2], max(dp[n - 1][3], dp[n - 1][4]))
        
        // 另一种解法
//        guard prices.count > 0 else {
//            return 0
//        }
//
//        let n = prices.count
//        var minNum1 = prices[0]
//        var minNum2 = minNum1
//        var res1 = 0
//        var res2 = 0
//        for i in 1 ..< n {
//            minNum1 = min(minNum1, prices[i])
//            res1 = max(res1, prices[i] - minNum1)
//            // 第二次买的时候，用第一次赚取到的利润进行补贴：prices[i] - res1
//            minNum2 = min(minNum2, prices[i] - res1)
//            res2 = max(res2, prices[i] - minNum2)
//        }
//        return res2
    }
    
    /**
     188. 买卖股票的最佳时机 IV
     
     给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

     设计一个算法来计算你所能获取的最大利润。你最多可以完成 k 笔交易。

     注意: 你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

     示例 1:

     输入: [2,4,1], k = 2
     输出: 2
     解释: 在第 1 天 (股票价格 = 2) 的时候买入，在第 2 天 (股票价格 = 4) 的时候卖出，这笔交易所能获得利润 = 4-2 = 2 。
     示例 2:

     输入: [3,2,6,5,0,3], k = 2
     输出: 7
     解释: 在第 2 天 (股票价格 = 2) 的时候买入，在第 3 天 (股票价格 = 6) 的时候卖出, 这笔交易所能获得利润 = 6-2 = 4 。
          随后，在第 5 天 (股票价格 = 0) 的时候买入，在第 6 天 (股票价格 = 3) 的时候卖出, 这笔交易所能获得利润 = 3-0 = 3 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit4(_ k: Int, _ prices: [Int]) -> Int {
        guard k > 0 && prices.count > 1 else {
            return 0
        }
        
        // k >= prices.count/2 时，相当于可以无限买卖，与 “122. 买卖股票的最佳时机 II” 解法相同
        if k >= prices.count/2 {
            var res = 0
            for i in 1 ..< prices.count {
                if prices[i] > prices[i - 1] {
                    res += (prices[i] - prices[i - 1])
                }
            }
            return res
        } else {
            var minNums = [Int](repeating: prices[0], count: k)
            var ress = [Int](repeating: 0, count: k)
            for i in 1 ..< prices.count {
                // 用前一次的利润补贴后一次的买入
                for j in 0 ..< k {
                    if 0 == j {
                        minNums[j] = min(minNums[j], prices[i])
                    } else {
                        minNums[j] = min(minNums[j], prices[i] - ress[j - 1])
                    }
                    ress[j] = max(ress[j], prices[i] - minNums[j])
                }
            }
            return ress[k - 1]
        }
    }
    
    /**
     309. 最佳买卖股票时机含冷冻期
     
     给定一个整数数组，其中第 i 个元素代表了第 i 天的股票价格 。

     设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:

     你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
     卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
     示例:

     输入: [1,2,3,0,2]
     输出: 3
     解释: 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit5(_ prices: [Int]) -> Int {
        let n = prices.count
        guard n > 1 else {
            return 0
        }
        
        // 动态规划 + 状态机，dp[i][j] 表示 [0, i] 区间内，在下标为 i 这一天状态为 j 时的最大收益
        // 3 种状态：0 - 不持股(卖)，1 - 持股(买)，2 - 冷冻期
        var dp = [[Int]]()
        for _ in 0 ..< n {
            dp.append([Int](repeating: 0, count: 3))
        }
        
        // 初始化
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[0][2] = 0
        
        for i in 1 ..< n {
            // 不持股 -> 不持股，或者持股 -> 不持股
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            // 持股 -> 持股，或者冷冻期 -> 持股
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][2] - prices[i])
            // 不持股 -> 冷冻期
            dp[i][2] = dp[i - 1][0]
        }
        
        return max(dp[n - 1][0], dp[n - 1][2])
    }
    
    /**
     714. 买卖股票的最佳时机含手续费
     
     给定一个整数数组 prices，其中第 i 个元素代表了第 i 天的股票价格 ；非负整数 fee 代表了交易股票的手续费用。

     你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，在卖出它之前你就不能再继续购买股票了。

     返回获得利润的最大值。

     注意：这里的一笔交易指买入持有并卖出股票的整个过程，每笔交易你只需要为支付一次手续费。

     示例 1:

     输入: prices = [1, 3, 2, 8, 4, 9], fee = 2
     输出: 8
     解释: 能够达到的最大利润:
     在此处买入 prices[0] = 1
     在此处卖出 prices[3] = 8
     在此处买入 prices[4] = 4
     在此处卖出 prices[5] = 9
     总利润: ((8 - 1) - 2) + ((9 - 4) - 2) = 8.
     注意:

     0 < prices.length <= 50000.
     0 < prices[i] < 50000.
     0 <= fee < 50000.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxProfit(_ prices: [Int], _ fee: Int) -> Int {
        guard prices.count > 1 else {
            return 0
        }
        
        // 不持有股票时的最大利润
        var cash = 0
        // 持有股票时的最大利润
        var hold = -prices[0]
        for i in 1 ..< prices.count {
            cash = max(cash, hold + prices[i] - fee)
            hold = max(hold, cash - prices[i])
        }
        return cash
    }
    
    /**
     279. 完全平方数
     
     给定正整数 n，找到若干个完全平方数（比如 1, 4, 9, 16, ...）使得它们的和等于 n。你需要让组成和的完全平方数的个数最少。

     示例 1:

     输入: n = 12
     输出: 3
     解释: 12 = 4 + 4 + 4.
     示例 2:

     输入: n = 13
     输出: 2
     解释: 13 = 4 + 9.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/perfect-squares
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numSquares(_ n: Int) -> Int {
        // 先找到平方数的数组，后面的跟零钱兑换问题一样的解法
        let squares = self.squares(n)
        guard squares.count > 0 else {
            return -1
        }
        
        var res = Int.max
        dfsNumSquares(n, 0, 0, squares, &res)
        return Int.max == res ? -1 : res
    }
    
    private func dfsNumSquares(_ n: Int, _ index: Int, _ count: Int, _ squares: [Int], _ res: inout Int) {
        // terminator
        if 0 == n {
            res = min(res, count)
            return
        }
        
        if index == squares.count {
            return
        }
        
        // process current logic
        var k = n/squares[index]
        
        // drill down
        while k >= 0 && k + count < res {
            dfsNumSquares(n - k * squares[index], index + 1, count + k, squares, &res)
            k -= 1
        }
        
        // restore
    }
    
    private func squares(_ n: Int) -> [Int] {
        if n <= 1 {
            return 0 == n ? [Int]() : [1]
        }
        
        var res = [Int]()
        // 先求 n 的平方根
        var sqrtN = n
        while sqrtN * sqrtN > n {
            sqrtN = (sqrtN + n/sqrtN)/2
        }
        res.append(sqrtN * sqrtN)
        sqrtN -= 1
        while sqrtN > 0 {
            if sqrtN * sqrtN < n {
                res.append(sqrtN * sqrtN)
            }
            sqrtN -= 1
        }
        return res
    }
    
    /**
     10 月 10 日 每日一题推荐：
     
     74. 搜索二维矩阵
     
     编写一个高效的算法来判断 m x n 矩阵中，是否存在一个目标值。该矩阵具有如下特性：

     每行中的整数从左到右按升序排列。
     每行的第一个整数大于前一行的最后一个整数。
     示例 1:

     输入:
     matrix = [
       [1,   3,  5,  7],
       [10, 11, 16, 20],
       [23, 30, 34, 50]
     ]
     target = 3
     输出: true
     示例 2:

     输入:
     matrix = [
       [1,   3,  5,  7],
       [10, 11, 16, 20],
       [23, 30, 34, 50]
     ]
     target = 13
     输出: false

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/search-a-2d-matrix
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func searchMatrix(_ matrix: [[Int]], _ target: Int) -> Bool {
        let m = matrix.count
        guard m > 0 else {
            return false
        }
        
        let n = matrix[0].count
        guard n > 0 else {
            return false
        }
        
        // 从右上角开始搜索
        var i = 0, j = n - 1
        while i < m && j >= 0 {
            if target == matrix[i][j] {
                return true
            } else if target > matrix[i][j] {   // 丢掉第 i 行
                i += 1
            } else if target < matrix[i][j] {   // 丢掉第 j 列
                j -= 1
            }
        }
        
        return false
    }
    
    /**
     72. 编辑距离
     
     给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作数 。

     你可以对一个单词进行如下三种操作：

     插入一个字符
     删除一个字符
     替换一个字符
      

     示例 1：

     输入：word1 = "horse", word2 = "ros"
     输出：3
     解释：
     horse -> rorse (将 'h' 替换为 'r')
     rorse -> rose (删除 'r')
     rose -> ros (删除 'e')
     示例 2：

     输入：word1 = "intention", word2 = "execution"
     输出：5
     解释：
     intention -> inention (删除 't')
     inention -> enention (将 'i' 替换为 'e')
     enention -> exention (将 'n' 替换为 'x')
     exention -> exection (将 'n' 替换为 'c')
     exection -> execution (插入 'u')

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/edit-distance
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minDistance(_ word1: String, _ word2: String) -> Int {
        let m = word1.count
        guard m > 0 else {
            return word2.count
        }
        
        let n = word2.count
        guard n > 0 else {
            return m
        }
        
        // dp[i][j] 表示 word1[1 ... i] 转换成 word2[1 ... j] 所需的最少操作数(这里假设下标从 1 开始，实际操作时，索引减 1 即可)
        var dp = [[Int]]()
        for _ in 0 ... m {
            dp.append([Int](repeating: 0, count: n + 1))
        }
        
        // 初始化
        for i in 1 ... m {
            dp[i][0] = i
        }
        
        for i in 1 ... n {
            dp[0][i] = i
        }
        
        
        let word1Arr = Array(word1)
        let word2Arr = Array(word2)
        for i in 1 ... m {
            for j in 1 ... n {
                if word1Arr[i - 1] != word2Arr[j - 1] {
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                } else {
                    dp[i][j] = dp[i - 1][j - 1]
                }
            }
        }
        
        return dp[m][n]
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
        guard nums.count > 1 else {
            return true
        }
        
        var endPosition = nums.count - 1
        for i in (0 ... nums.count - 2).reversed() {
            if i + nums[i] >= endPosition {
                endPosition = i
            }
        }
        
        return 0 == endPosition
    }
    
    /**
     45. 跳跃游戏 II
     
     给定一个非负整数数组，你最初位于数组的第一个位置。

     数组中的每个元素代表你在该位置可以跳跃的最大长度。

     你的目标是使用最少的跳跃次数到达数组的最后一个位置。

     示例:

     输入: [2,3,1,1,4]
     输出: 2
     解释: 跳到最后一个位置的最小跳跃数是 2。
          从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
     说明:

     假设你总是可以到达数组的最后一个位置。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/jump-game-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func jump(_ nums: [Int]) -> Int {
        var maxPosition = 0 // 能跳到的最远位置
        var end = 0         // 下标更新到该最远位置
        var steps = 0       // 步数
        for i in 0 ..< nums.count - 1 {
            // 找能跳的最远的
            maxPosition = max(maxPosition, nums[i] + i)
            if i == end { // 遇到边界，就更新边界，并且步数加一
                end = maxPosition
                steps += 1
            }
        }
        return steps
    }
    
    /**
     10 月 11 日每日一题推荐
     
     剑指 Offer 05. 替换空格
     
     请实现一个函数，把字符串 s 中的每个空格替换成"%20"。

      

     示例 1：

     输入：s = "We are happy."
     输出："We%20are%20happy."
      

     限制：

     0 <= s 的长度 <= 10000

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/ti-huan-kong-ge-lcof
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func replaceSpace(_ s: String) -> String {
        var spaceCount = 0
        for char in s {
            if char == " " {
                spaceCount += 1
            }
        }
        
        let count = s.count + 2 * spaceCount
        var arr = [Character](repeating: " ", count: count)
        var index = 0
        for char in s {
            if char == " " {
                arr[index] = "%"
                index += 1
                arr[index] = "2"
                index += 1
                arr[index] = "0"
                index += 1
            } else {
                arr[index] = char
                index += 1
            }
        }
        return arr.compactMap { "\($0)" }.joined()
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
        
        // 动态规划解法
        // dp[i][j] 表示从左上角到点 (i, j) 的最小路径和
//        var dp = [[Int]]()
//        for _ in 0 ..< m {
//            dp.append([Int](repeating: 0, count: n))
//        }
//
//        dp[0][0] = grid[0][0]
//
//        for i in 1 ..< m {
//            dp[i][0] = dp[i - 1][0] + grid[i][0]
//        }
//
//        for i in 1 ..< n {
//            dp[0][i] = dp[0][i - 1] + grid[0][i]
//        }
//
//        for i in 1 ..< m {
//            for j in 1 ..< n {
//                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
//            }
//        }
//
//        return dp[m - 1][n - 1]
        
        // 递归解法
        var memo = [String : Int]()
        return dfsMinPathSum(grid, m, n, 0, 0, &memo)
    }
    
    private func dfsMinPathSum(_ grid: [[Int]], _ m: Int, _ n: Int, _ row: Int, _ col: Int, _ memo: inout [String : Int]) -> Int {
        // terminator
        if row >= m || col >= n {
            return Int.max
        }
        
        let key = String(row) + "-" + String(col)
        if let val = memo[key] {
            return val
        }
        
        let sum = min(dfsMinPathSum(grid, m, n, row + 1, col, &memo), dfsMinPathSum(grid, m, n, row, col + 1, &memo))
        memo[key] = grid[row][col] + (Int.max == sum ? 0 : sum)
        return memo[key]!
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
        // 参考链接：https://leetcode-cn.com/problems/decode-ways/solution/c-wo-ren-wei-hen-jian-dan-zhi-guan-de-jie-fa-by-pr/
        let n = s.count
        guard n > 0 else {
            return 0
        }
        
        let sArr = Array(s)
        if "0" == sArr[0] { // 0 开头的无法解码
            return 0
        }
        
        // dp[i] 为 s[0...i] 的译码方法总数
        var dp = [Int](repeating: 0, count: n)
        dp[0] = 1
        for i in 1 ..< n {
            if "0" != sArr[i] {
                dp[i] = dp[i - 1]
            }
            
            let pre = Int(String(sArr[i - 1]))!
            let current = Int(String(sArr[i]))!
            if 10 * pre + current >= 10 && 10 * pre + current <= 26 {
                if 1 == i {
                    dp[i] = dp[i] + 1
                } else {
                    // s[i - 1] 与 s[i] 分开译码，为 dp[i - 1]，合并译码为 dp[i - 2]
                    dp[i] = dp[i] + dp[i - 2]
                }
            }
        }
        return dp[n - 1]
    }
    
    /**
     221. 最大正方形
     
     在一个由 0 和 1 组成的二维矩阵内，找到只包含 1 的最大正方形，并返回其面积。

     示例:

     输入:

     1 0 1 0 0
     1 0 1 1 1
     1 1 1 1 1
     1 0 0 1 0

     输出: 4

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximal-square
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maximalSquare(_ matrix: [[Character]]) -> Int {
        let m = matrix.count
        guard m > 0 else {
            return 0
        }
        
        let n = matrix[0].count
        guard n > 0 else {
            return 0
        }
        
        // dp[i][j] 表示以点 (i, j) 为右下角的只包含 1 的最大正方形的边长
        var dp = [[Int]]()
        for _ in 0 ..< m {
            dp.append([Int](repeating: 0, count: n))
        }
        
        var maxSide = 0
        for i in 0 ..< m {
            for j in 0 ..< n {
                if "1" == matrix[i][j] {
                    if 0 == i || 0 == j {
                        dp[i][j] = 1
                    } else {
                        dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    }
                }
                maxSide = max(maxSide, dp[i][j])
            }
        }
        
        return maxSide * maxSide
    }
    
    /**
     24. 两两交换链表中的节点
     
     给定一个链表，两两交换其中相邻的节点，并返回交换后的链表。

     你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

      

     示例 1：


     输入：head = [1,2,3,4]
     输出：[2,1,4,3]
     示例 2：

     输入：head = []
     输出：[]
     示例 3：

     输入：head = [1]
     输出：[1]
      

     提示：

     链表中节点的数目在范围 [0, 100] 内
     0 <= Node.val <= 100

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/swap-nodes-in-pairs
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func swapPairs(_ head: ListNode?) -> ListNode? {
        if nil == head || nil == head?.next {
            return head
        }
        
        let dummy = ListNode()
        var node = head
        while nil != node {
            let next = node?.next
            let next1 = next?.next
            let next2 = next1?.next
            next?.next = node
            node?.next = nil == next2 ? next1 : next2
            node = next1
            if nil == dummy.next {
                dummy.next = next
            }
        }
        return dummy.next
    }
    
    func swapPairsRecursive(_ head: ListNode?) -> ListNode? {
        if nil == head || nil == head?.next {
            return head
        }
        
        let next = head?.next
        head?.next = swapPairsRecursive(next?.next)
        next?.next = head
        return next
    }
    
    /**
     647. 回文子串
     
     给定一个字符串，你的任务是计算这个字符串中有多少个回文子串。

     具有不同开始位置或结束位置的子串，即使是由相同的字符组成，也会被视作不同的子串。

      

     示例 1：

     输入："abc"
     输出：3
     解释：三个回文子串: "a", "b", "c"
     示例 2：

     输入："aaa"
     输出：6
     解释：6个回文子串: "a", "a", "a", "aa", "aa", "aaa"
      

     提示：

     输入的字符串长度不会超过 1000 。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/palindromic-substrings
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func countSubstrings(_ s: String) -> Int {
        let sArr = Array(s)
        let n = sArr.count
        guard n > 1 else {
            return n
        }
        
        // dp[i][j] 表示 s[i ... j ] 是否为回文串
        var dp = [[Bool]]()
        for _ in 0 ..< n {
            dp.append([Bool](repeating: false, count: n))
        }
        
        var res = 0
        for j in 0 ..< n {
            for i in 0 ... j {
                if sArr[i] == sArr[j] {
                    if i + 1 < n && j - 1 >= 0 && dp[i + 1][j - 1] {
                        dp[i][j] = true
                        res += 1
                    } else if j - i < 2 {
                        dp[i][j] = true
                        res += 1
                    }
                }
            }
        }
        return res
    }
    
    /**
     621. 任务调度器
     
     给定一个用字符数组表示的 CPU 需要执行的任务列表。其中包含使用大写的 A - Z 字母表示的26 种不同种类的任务。任务可以以任意顺序执行，并且每个任务都可以在 1 个单位时间内执行完。CPU 在任何一个单位时间内都可以执行一个任务，或者在待命状态。

     然而，两个相同种类的任务之间必须有长度为 n 的冷却时间，因此至少有连续 n 个单位时间内 CPU 在执行不同的任务，或者在待命状态。

     你需要计算完成所有任务所需要的最短时间。

      

     示例 ：

     输入：tasks = ["A","A","A","B","B","B"], n = 2
     输出：8
     解释：A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
          在本示例中，两个相同类型任务之间必须间隔长度为 n = 2 的冷却时间，而执行一个任务只需要一个单位时间，所以中间出现了（待命）状态。
      

     提示：

     任务的总个数为 [1, 10000]。
     n 的取值范围为 [0, 100]。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/task-scheduler
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private let letters: [Character] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    func leastInterval(_ tasks: [Character], _ n: Int) -> Int {
        guard n > 0 else {
            return tasks.count
        }
        
        var dict = [Int]()
        for _ in 0 ..< 26 {
            dict.append(0)
        }
        
        let A: Character = "A"
        for task in tasks {
            dict[Int(task.asciiValue! - A.asciiValue!)] += 1
        }
        dict.sort()
        
        var res = 0
        while dict[25] > 0 {
            var i = 0
            while i <= n {
                if 0 == dict[25] {
                    break
                }
                
                if i < 26 && dict[25 - i] > 0 {
                    dict[25 - i] -= 1
                }
                
                res += 1
                i += 1
            }
            dict.sort()
        }
        return res
    }
    
    /**
     1002. 查找常用字符
     
     给定仅有小写字母组成的字符串数组 A，返回列表中的每个字符串中都显示的全部字符（包括重复字符）组成的列表。例如，如果一个字符在每个字符串中出现 3 次，但不是 4 次，则需要在最终答案中包含该字符 3 次。

     你可以按任意顺序返回答案。

      

     示例 1：

     输入：["bella","label","roller"]
     输出：["e","l","l"]
     示例 2：

     输入：["cool","lock","cook"]
     输出：["c","o"]
      

     提示：

     1 <= A.length <= 100
     1 <= A[i].length <= 100
     A[i][j] 是小写字母

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/find-common-characters
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private let lowerLetters: [Character] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    func commonChars(_ A: [String]) -> [String] {
        guard A.count > 0 else {
            return [String]()
        }
        
        var minDict = [Character : Int]()
        for letter in lowerLetters {
            minDict[letter] = Int.max
        }
        
        for string in A {
            var dict = [Character : Int]()
            for letter in lowerLetters {
                dict[letter] = 0
            }
            
            for char in Array(string) {
                dict[char] = dict[char]! + 1
            }
            
            for letter in lowerLetters {
                minDict[letter] = min(minDict[letter]!, dict[letter]!)
            }
        }
        
        var res = [String]()
        for letter in lowerLetters {
            let count = minDict[letter]!
            if count > 0 {
                for _ in 0 ..< count {
                    res.append(String(letter))
                }
            }
        }
        
        return res
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
        guard s.count > 1 else {
            return 0
        }
        
        let sArr = Array(s)
        /**
         dp[i] 表示以下标为 i 的字符结尾的最长有效括号长度
         
         参考链接：https://leetcode-cn.com/problems/longest-valid-parentheses/solution/zui-chang-you-xiao-gua-hao-by-leetcode-solution/
         */
        var dp = [Int](repeating: 0, count: sArr.count)
        var res = 0
        for i in 1 ..< sArr.count {
            if ")" == sArr[i] {
                if "(" == sArr[i - 1] {
                    dp[i] = i >= 2 ? dp[i - 2] + 2 : 2
                } else if i - dp[i - 1] > 0 && "(" == sArr[i - dp[i - 1] - 1] {
                    dp[i] = i - dp[i - 1] > 1 ? dp[i - 1] + dp[i - dp[i - 1] - 2] + 2 : dp[i - 1] + 2
                }
            }
            res = max(res, dp[i])
        }
        return res
        
        // 暴力法，时间复杂度为 O(n^3)，超时
//        guard s.count > 1 else {
//            return 0
//        }
//
//        let sArr = Array(s)
//        var res = 0
//        for i in 1 ..< sArr.count {
//            for j in 0 ... i {
//                if isValidParentheses(Array(sArr[j ... i])) {
//                    res = max(res, i - j + 1)
//                }
//            }
//        }
//        return res
    }
    
    private func isValidParentheses(_ s: [Character]) -> Bool {
        guard s.count > 1 && 0 == s.count%2 else {
            return false
        }
        
        if s[0] == ")" {
            return false
        }
        
        var stack = [Character]()
        stack.append(s[0])
        for i in 1 ..< s.count {
            if "(" == s[i] {
                stack.append("(")
            } else {
                if !stack.isEmpty {
                    stack.removeLast()
                } else {
                    return false
                }
            }
        }
        
        return stack.isEmpty
    }
    
    /**
     363. 矩形区域不超过 K 的最大数值和
     
     给定一个非空二维矩阵 matrix 和一个整数 k，找到这个矩阵内部不大于 k 的最大矩形和。

     示例:

     输入: matrix = [[1,0,1],[0,-2,3]], k = 2
     输出: 2
     解释: 矩形区域 [[0, 1], [-2, 3]] 的数值和是 2，且 2 是不超过 k 的最大数字（k = 2）。
     说明：

     矩阵内的矩形区域面积必须大于 0。
     如果行数远大于列数，你将如何解答呢？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxSumSubmatrix(_ matrix: [[Int]], _ k: Int) -> Int {
        let m = matrix.count
        guard m > 0 else {
            return 0
        }
        
        let n = matrix[0].count
        guard n > 0 else {
            return 0
        }
        
        var res = Int.min
        for left in 0 ..< n {
            // 以 left 为左边界，每行的总和
            var sum = [Int](repeating: 0, count: m)
            for right in left ..< n {
                for j in 0 ..< m {
                    sum[j] += matrix[j][right]
                }
                print(" ============================= \n")
                // left、right 为边界下的矩阵，求不超过 k 的最大数值和
                // 求最大数值和，使用前缀和数组，参考链接：https://leetcode-cn.com/problems/max-sum-of-rectangle-no-larger-than-k/solution/qian-zhui-he-pai-xu-jin-ke-neng-tong-su-yi-dong-de/
                var prefixSum = [Int]()
                prefixSum.append(0)
                var cur = 0
                for num in sum {
                    cur += num
                    let loc = bisectLeft(prefixSum, cur - k)
                    print("----------- num: \(num), loc: \(loc), cur: \(cur) ----------- \n")
                    if prefixSum.count > loc {
                        res = max(res, cur - prefixSum[loc])
                    }
                    inorder(&prefixSum, cur)
                }
            }
        }
        return res
    }
    
    private func bisectLeft(_ arr: [Int], _ num: Int) -> Int {
        guard arr.count > 0 else {
            return 0
        }
        
        if num < arr[0] {
            return 0
        }
        
        if num > arr[arr.count - 1] {
            return arr.count
        }
        
        var left = 0
        var right = arr.count
        while left < right {
            let middle = left + (right - left)/2
            if arr[middle] < num {
                left = middle + 1
            } else {
                right = middle
            }
        }
        return left
    }
    
    private func inorder(_ arr: inout [Int], _ num: Int) {
        let index = bisectLeft(arr, num)
        arr.insert(num, at: index)
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
        preorderHelper(root, &res)
        return res
    }
    
    private func preorderHelper(_ root: Node?, _ res: inout [Int]) {
        if let val = root?.val {
            res.append(val)
        }
        
        if let children = root?.children {
            for child in children {
                preorderHelper(child, &res)
            }
        }
    }
    
    /**
     18. 四数之和
     
     给定一个包含 n 个整数的数组 nums 和一个目标值 target，判断 nums 中是否存在四个元素 a，b，c 和 d ，使得 a + b + c + d 的值与 target 相等？找出所有满足条件且不重复的四元组。

     注意：

     答案中不可以包含重复的四元组。

     示例：

     给定数组 nums = [1, 0, -1, 0, -2, 2]，和 target = 0。

     满足要求的四元组集合为：
     [
       [-1,  0, 0, 1],
       [-2, -1, 1, 2],
       [-2,  0, 0, 2]
     ]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/4sum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        // 参考链接：https://leetcode-cn.com/problems/4sum/solution/java-jian-dan-yi-dong-ji-hu-shuang-bai-ji-bai-si-s/
        var res = [[Int]]()
        guard nums.count >= 4 else {
            return res
        }
        
        let sortedNums = nums.sorted(by: <)
        let count = sortedNums.count
        for i in 0 ..< count - 3 {
            if i > 0 && sortedNums[i] == sortedNums[i - 1] { // 去重
                continue
            }
            
            var curMin = sortedNums[i] + sortedNums[i + 1] + sortedNums[i + 2] + sortedNums[i + 3]
            if curMin > target {
                break
            }
            
            var curMax = sortedNums[i] + sortedNums[count - 1] + sortedNums[count - 2] + sortedNums[count - 3]
            if curMax < target {
                continue
            }
                        
            for j in i + 1 ..< count - 2 {
                if j > i + 1 && sortedNums[j] == sortedNums[j - 1] { // 去重
                    continue
                }
                
                if j + 2 < count {
                    curMin = sortedNums[i] + sortedNums[j] + sortedNums[j + 1] + sortedNums[j + 2]
                    if curMin > target {
                        break
                    }
                }
                
                if count - 2 > j {
                    curMax = sortedNums[i] + sortedNums[j] + sortedNums[count - 1] + sortedNums[count - 2]
                    if curMax < target {
                        continue
                    }
                }
                
                var k = j + 1
                var m = count - 1
                while k < m {
                    let curSum = sortedNums[i] + sortedNums[j] + sortedNums[k] + sortedNums[m]
                    if target == curSum {
                        res.append([sortedNums[i], sortedNums[j], sortedNums[k], sortedNums[m]])
                        
                        k += 1
                        while k < m && sortedNums[k] == sortedNums[k - 1] {
                            k += 1
                        }
                        
                        m -= 1
                        while k < m && sortedNums[m] == sortedNums[m + 1] {
                            m -= 1
                        }
                    } else if target > curSum {
                        k += 1
                        while k < m && sortedNums[k] == sortedNums[k - 1] {
                            k += 1
                        }
                    } else {
                        m -= 1
                        while k < m && sortedNums[m] == sortedNums[m + 1] {
                            m -= 1
                        }
                    }
                }
            }
        }
        return res
    }
    
    /**
     116. 填充每个节点的下一个右侧节点指针
     
     给定一个完美二叉树，其所有叶子节点都在同一层，每个父节点都有两个子节点。二叉树定义如下：

     struct Node {
       int val;
       Node *left;
       Node *right;
       Node *next;
     }
     填充它的每个 next 指针，让这个指针指向其下一个右侧节点。如果找不到下一个右侧节点，则将 next 指针设置为 NULL。

     初始状态下，所有 next 指针都被设置为 NULL。

      

     示例：



     输入：{"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":null,"right":null,"val":4},"next":null,"right":{"$id":"4","left":null,"next":null,"right":null,"val":5},"val":2},"next":null,"right":{"$id":"5","left":{"$id":"6","left":null,"next":null,"right":null,"val":6},"next":null,"right":{"$id":"7","left":null,"next":null,"right":null,"val":7},"val":3},"val":1}

     输出：{"$id":"1","left":{"$id":"2","left":{"$id":"3","left":null,"next":{"$id":"4","left":null,"next":{"$id":"5","left":null,"next":{"$id":"6","left":null,"next":null,"right":null,"val":7},"right":null,"val":6},"right":null,"val":5},"right":null,"val":4},"next":{"$id":"7","left":{"$ref":"5"},"next":null,"right":{"$ref":"6"},"val":3},"right":{"$ref":"4"},"val":2},"next":null,"right":{"$ref":"7"},"val":1}

     解释：给定二叉树如图 A 所示，你的函数应该填充它的每个 next 指针，以指向其下一个右侧节点，如图 B 所示。
      

     提示：

     你只能使用常量级额外空间。
     使用递归解题也符合要求，本题中递归程序占用的栈空间不算做额外的空间复杂度。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func connect(_ root: CompleteNode?) -> CompleteNode? {
        if nil == root {
            return nil
        }
        
        connectHelper(root!.left, root!.right)
        return root
    }
    
    private func connectHelper(_ left: CompleteNode?, _ right: CompleteNode?) {
        if nil == left || nil == right {
            return
        }
        
        left?.next = right
        connectHelper(left?.left, left?.right)
        connectHelper(right?.left, right?.right)
        connectHelper(left?.right, right?.left)
    }
    
    /**
     977. 有序数组的平方
     
     给定一个按非递减顺序排序的整数数组 A，返回每个数字的平方组成的新数组，要求也按非递减顺序排序。

      

     示例 1：

     输入：[-4,-1,0,3,10]
     输出：[0,1,9,16,100]
     示例 2：

     输入：[-7,-3,2,3,11]
     输出：[4,9,9,49,121]
      

     提示：

     1 <= A.length <= 10000
     -10000 <= A[i] <= 10000
     A 已按非递减顺序排序。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/squares-of-a-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func sortedSquares(_ A: [Int]) -> [Int] {
        // 双指针法
        var n = A.count
        var res = [Int](repeating: 0, count: n)
        var i = 0, j = n - 1
        while i <= j {
            let pre = A[i] * A[i]
            let next = A[j] * A[j]
            if pre > next {
                res[n - 1] = pre
                i += 1
            } else {
                res[n - 1] = next
                j -= 1
            }
            n -= 1
        }
        return res
//        var res = [Int]()
//        if A.count > 0 {
//            res.append(A[0] * A[0])
//            for i in 1 ..< A.count {
//                let num = A[i] * A[i]
//                let index = bisectLeft(res, num)
//                res.insert(num, at: index)
//            }
//        }
//        return res
    }
    
    /**
     33. 搜索旋转排序数组
     
     给你一个升序排列的整数数组 nums ，和一个整数 target 。

     假设按照升序排序的数组在预先未知的某个点上进行了旋转。（例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] ）。

     请你在数组中搜索 target ，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

      
     示例 1：

     输入：nums = [4,5,6,7,0,1,2], target = 0
     输出：4
     示例 2：

     输入：nums = [4,5,6,7,0,1,2], target = 3
     输出：-1
     示例 3：

     输入：nums = [1], target = 0
     输出：-1
      

     提示：

     1 <= nums.length <= 5000
     -10^4 <= nums[i] <= 10^4
     nums 中的每个值都 独一无二
     nums 肯定会在某个点上旋转
     -10^4 <= target <= 10^4

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func search(_ nums: [Int], _ target: Int) -> Int {
        guard nums.count > 0 else {
            return -1
        }
        
        var left = 0, right = nums.count - 1
        while left <= right {
            let middle = left + (right - left) >> 1
            if target == nums[middle] {
                return middle
            }
            
            if nums[middle] >= nums[left] { // 左侧递增段
                if target >= nums[left] && target < nums[middle] {
                    right = middle - 1
                } else {
                    left = middle + 1
                }
            } else {
                if target > nums[middle] && target <= nums[right] {
                    left = middle + 1
                } else {
                    right = middle - 1
                }
            }
        }
        
        return -1
    }
    
    /**
     52. N皇后 II
     
     n 皇后问题研究的是如何将 n 个皇后放置在 n×n 的棋盘上，并且使皇后彼此之间不能相互攻击。



     上图为 8 皇后问题的一种解法。

     给定一个整数 n，返回 n 皇后不同的解决方案的数量。

     示例:

     输入: 4
     输出: 2
     解释: 4 皇后问题存在如下两个不同的解法。
     [
      [".Q..",  // 解法 1
       "...Q",
       "Q...",
       "..Q."],

      ["..Q.",  // 解法 2
       "Q...",
       "...Q",
       ".Q.."]
     ]
      

     提示：

     皇后，是国际象棋中的棋子，意味着国王的妻子。皇后只做一件事，那就是“吃子”。当她遇见可以吃的棋子时，就迅速冲上去吃掉棋子。当然，她横、竖、斜都可走一或 N-1 步，可进可退。（引用自 百度百科 - 皇后 ）


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/n-queens-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func totalNQueens(_ n: Int) -> Int {
        guard n > 1 else {
            return n
        }
        
        var res = 0
        var board = [[Character]]()
        for _ in 0 ..< n {
            board.append([Character](repeating: ".", count: n))
        }
        totalNQueensBacktrack(&board, 0, &res)
        return res
    }
    
    private func totalNQueensBacktrack(_ board: inout [[Character]], _ row: Int, _ res: inout Int) {
        if row == board.count {
            res += 1
            return
        }
        
        let n = board[0].count
        for col in 0 ..< n {
            if !isValid(board, row, col) {
                continue
            }
            
            board[row][col] = "Q"
            totalNQueensBacktrack(&board, row + 1, &res)
            board[row][col] = "."
        }
    }
    
    private func isValid(_ board: [[Character]], _ row: Int, _ col: Int) -> Bool {
        let n = board[0].count
        // 行
        for i in 0 ..< n {
            if "Q" == board[i][col] {
                return false
            }
        }

        // 左上角
        var tempRow = row - 1, tempCol = col - 1
        while tempRow >= 0 && tempCol >= 0 {
            if "Q" == board[tempRow][tempCol] {
                return false
            }
            tempRow -= 1
            tempCol -= 1
        }

        // 右上角
        tempRow = row - 1
        tempCol = col + 1
        while tempRow >= 0 && tempCol < n {
            if "Q" == board[tempRow][tempCol] {
                return false
            }
            tempRow -= 1
            tempCol += 1
        }
        
        return true
    }
    
    /**
     410. 分割数组的最大值
     
     给定一个非负整数数组和一个整数 m，你需要将这个数组分成 m 个非空的连续子数组。设计一个算法使得这 m 个子数组各自和的最大值最小。

     注意:
     数组长度 n 满足以下条件:

     1 ≤ n ≤ 1000
     1 ≤ m ≤ min(50, n)
     示例:

     输入:
     nums = [7,2,5,10,8]
     m = 2

     输出:
     18

     解释:
     一共有四种方法将nums分割为2个子数组。
     其中最好的方式是将其分为[7,2,5] 和 [10,8]，
     因为此时这两个子数组各自的和的最大值为18，在所有情况中最小。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/split-array-largest-sum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func splitArray(_ nums: [Int], _ m: Int) -> Int {
        guard m > 0 else {
            return 0
        }
        
        var maxNum = 0
        var sum = 0
        for num in nums {
            maxNum = max(maxNum, num)
            sum += num
        }
        
        var left = maxNum
        var right = sum
        while left < right {
            let mid = left + (right - left)/2
            let splitCount = split(nums, mid)
            if splitCount > m { // 分割的数组太多，说明子数组和最大值较小
                left = mid + 1
            } else {
                right = mid
            }
        }
        return left
    }
    
    private func split(_ nums: [Int], _ maxNum: Int) -> Int {
        var split = 1
        var curSum = 0
        for num in nums {
            if curSum + num > maxNum {
                curSum = 0
                split += 1
            }
            curSum += num
        }
        return split
    }
    
    /**
     403. 青蛙过河
     
     一只青蛙想要过河。 假定河流被等分为 x 个单元格，并且在每一个单元格内都有可能放有一石子（也有可能没有）。 青蛙可以跳上石头，但是不可以跳入水中。

     给定石子的位置列表（用单元格序号升序表示）， 请判定青蛙能否成功过河（即能否在最后一步跳至最后一个石子上）。 开始时， 青蛙默认已站在第一个石子上，并可以假定它第一步只能跳跃一个单位（即只能从单元格1跳至单元格2）。

     如果青蛙上一步跳跃了 k 个单位，那么它接下来的跳跃距离只能选择为 k - 1、k 或 k + 1个单位。 另请注意，青蛙只能向前方（终点的方向）跳跃。

     请注意：

     石子的数量 ≥ 2 且 < 1100；
     每一个石子的位置序号都是一个非负整数，且其 < 231；
     第一个石子的位置永远是0。
     示例 1:

     [0,1,3,5,6,8,12,17]

     总共有8个石子。
     第一个石子处于序号为0的单元格的位置, 第二个石子处于序号为1的单元格的位置,
     第三个石子在序号为3的单元格的位置， 以此定义整个数组...
     最后一个石子处于序号为17的单元格的位置。

     返回 true。即青蛙可以成功过河，按照如下方案跳跃：
     跳1个单位到第2块石子, 然后跳2个单位到第3块石子, 接着
     跳2个单位到第4块石子, 然后跳3个单位到第6块石子,
     跳4个单位到第7块石子, 最后，跳5个单位到第8个石子（即最后一块石子）。
     示例 2:

     [0,1,2,3,4,8,9,11]

     返回 false。青蛙没有办法过河。
     这是因为第5和第6个石子之间的间距太大，没有可选的方案供青蛙跳跃过去。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/frog-jump
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func canCross(_ stones: [Int]) -> Bool {
        guard stones.count > 0 else {
            return false
        }
        
        var dict = [Int : Set<Int>]()
        for stone in stones {
            dict[stone] = Set<Int>()
        }
        let dictSet = Set(dict.keys)
        dict[0]?.insert(0)
        for stone in stones {
            let steps = dict[stone]!
            for step in steps {
                var i = step - 1
                while i <= step + 1 {
                    if i > 0 && dictSet.contains(i + stone) {
                        dict[i + stone]!.insert(i)
                    }
                    i += 1
                }
            }
        }
        return dict[stones[stones.count - 1]]!.count > 0
    }
    
    /**
     552. 学生出勤记录 II
     
     给定一个正整数 n，返回长度为 n 的所有可被视为可奖励的出勤记录的数量。 答案可能非常大，你只需返回结果mod 109 + 7的值。

     学生出勤记录是只包含以下三个字符的字符串：

     'A' : Absent，缺勤
     'L' : Late，迟到
     'P' : Present，到场
     如果记录不包含多于一个'A'（缺勤）或超过两个连续的'L'（迟到），则该记录被视为可奖励的。

     示例 1:

     输入: n = 2
     输出: 8
     解释：
     有8个长度为2的记录将被视为可奖励：
     "PP" , "AP", "PA", "LP", "PL", "AL", "LA", "LL"
     只有"AA"不会被视为可奖励，因为缺勤次数超过一次。
     注意：n 的值不会超过100000。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/student-attendance-record-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func checkRecord(_ n: Int) -> Int {
        // 动态规划
        guard n > 1 else {
            return 1 == n ? 3 : 0
        }
        
        let M = 1000000007
        var f = [Int]()
        if n <= 5 {
            for _ in 0 ..< 6 {
                f.append(0)
            }
        } else {
            for _ in 0 ..< n + 1 {
                f.append(0)
            }
        }
        
        f[0] = 1
        f[1] = 2
        f[2] = 4
        f[3] = 7
        if n >= 4 {
            for i in 4 ... n {
                f[i] = ((2 * f[i - 1]) % M + (M - f[i - 4])) % M
//                f[i] = (2 * f[i - 1]) % M - (f[i - 4] % M)
//                if f[i] < 0 {
//                    print("i: \(i), f[i]: \(f[i])")
//                }
            }
        }
        
        var res = f[n]
        for i in 1 ... n {
            res += (f[i - 1] * f[n - i])%M
        }
        return res%M
        
        // 暴力法，超时
//        var records = [Character]()
//        var totalRecords = [[Character]]()
//        recordBacktrack(n, &records, &totalRecords)
//        var count = 0
//        for records in totalRecords {
//            if isValidRecord(records) {
//                count = (count + 1)%1000000007
//            }
//        }
//        return count
    }
    
    private let recordChars: [Character] = ["A", "L", "P"]
    private func recordBacktrack(_ n: Int, _ records: inout [Character], _ res: inout [[Character]]) {
        if records.count == n {
            res.append(records)
            return
        }
        
        for record in recordChars {
            records.append(record)
            recordBacktrack(n, &records, &res)
            records.remove(at: records.count - 1)
        }
    }
    
    private func isValidRecord(_ arr: [Character]) -> Bool {
        var ACount = 0
        var LCount = 0
        var i = 0
        while i < arr.count {
            if "A" == arr[i] {
                ACount += 1
            } else if "L" == arr[i] {
                if i + 2 < arr.count && arr[i] == arr[i + 1] && arr[i] == arr[i + 2] {
                    LCount += 1
                }
            }
                        
            if ACount > 1 {
                return false
            }
            
            if LCount > 0 {
                return false
            }
            
            i += 1
        }
        
        return true
    }
    
    /**
     19. 删除链表的倒数第N个节点
     
     给定一个链表，删除链表的倒数第 n 个节点，并且返回链表的头结点。

     示例：

     给定一个链表: 1->2->3->4->5, 和 n = 2.

     当删除了倒数第二个节点后，链表变为 1->2->3->5.
     说明：

     给定的 n 保证是有效的。

     进阶：

     你能尝试使用一趟扫描实现吗？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        // 双指针法，只用一次遍历
        var firstNode = head
        let dummy = ListNode(0)
        dummy.next = head
        var secondNode: ListNode? = dummy
        for _ in 0 ..< n {
            firstNode = firstNode?.next
        }
        
        while firstNode != nil {
            firstNode = firstNode?.next
            secondNode = secondNode?.next
        }
        
        secondNode?.next = secondNode?.next?.next
        return dummy.next
//        var count = 0
//        var node = head
//        while node != nil {
//            count += 1
//            node = node?.next
//        }
//
//        guard count >= n else {
//            return head
//        }
//
//        let dummyHead = ListNode()
//        dummyHead.next = head
//        node = head
//        var pre: ListNode? = dummyHead
//        let toBeDeletedCount = count - n + 1
//        count = 1
//        while node != nil {
//            if toBeDeletedCount == count {
//                pre?.next = node?.next
//                break
//            }
//            pre = node
//            node = node?.next
//            count += 1
//        }
//        return dummyHead.next
    }
    
    /**
     212. 单词搜索 II
     
     给定一个二维网格 board 和一个字典中的单词列表 words，找出所有同时在二维网格和字典中出现的单词。

     单词必须按照字母顺序，通过相邻的单元格内的字母构成，其中“相邻”单元格是那些水平相邻或垂直相邻的单元格。同一个单元格内的字母在一个单词中不允许被重复使用。

     示例:

     输入:
     words = ["oath","pea","eat","rain"] and board =
     [
       ['o','a','a','n'],
       ['e','t','a','e'],
       ['i','h','k','r'],
       ['i','f','l','v']
     ]

     输出: ["eat","oath"]
     说明:
     你可以假设所有输入都由小写字母 a-z 组成。

     提示:

     你需要优化回溯算法以通过更大数据量的测试。你能否早点停止回溯？
     如果当前单词不存在于所有单词的前缀中，则可以立即停止回溯。什么样的数据结构可以有效地执行这样的操作？散列表是否可行？为什么？ 前缀树如何？如果你想学习如何实现一个基本的前缀树，请先查看这个问题： 实现Trie（前缀树）。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/word-search-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private var _board: [[Character]] = [[Character]]()
    private var _result = [String]()
    func findWords(_ board: [[Character]], _ words: [String]) -> [String] {
        let m = board.count
        guard m > 0 else {
            return _result
        }
        
        let n = board[0].count
        guard n > 0 else {
            return _result
        }
        
        // 根据 words 构造字典树
        let root = TrieNode()
        for word in words {
            var node = root
            for character in Array(word) {
                if node.children.keys.contains(character) {
                    node = node.children[character]!
                } else {
                    let newNode = TrieNode()
                    node.children[character] = newNode
                    node = newNode
                }
            }
            node.word = word
        }
        
        _board = board
        for i in 0 ..< m {
            for j in 0 ..< n {
                if root.children.keys.contains(board[i][j]) {
                    findWordsBacktrack(i, j, root)
                }
            }
        }
        
        return _result
    }
    
    private func findWordsBacktrack(_ row: Int, _ col: Int, _ parent: TrieNode) {
        let letter = _board[row][col]
        let currentNode = parent.children[letter]
        if nil != currentNode && nil != currentNode!.word {
            _result.append(currentNode!.word!)
            currentNode!.word = nil
        }
        
        _board[row][col] = "#"
        let rowOffset = [-1, 0, 1, 0]
        let colOffset = [0, 1, 0, -1]
        for i in 0 ..< 4 {
            let newRow = row + rowOffset[i]
            let newCol = col + colOffset[i]
            if newRow >= _board.count || newRow < 0 || newCol >= _board[0].count || newCol < 0 {
                continue
            }
            
            if nil != currentNode && currentNode!.children.keys.contains(_board[newRow][newCol]) {
                findWordsBacktrack(newRow, newCol, currentNode!)
            }
        }
        _board[row][col] = letter
        
        if nil != currentNode && currentNode!.children.isEmpty {
            currentNode!.children.removeValue(forKey: letter)
        }
    }
    
    private func dfsFindWords(_ board: [[Character]], _ wordSet: Set<String>, _ row: Int, _ col: Int, _ chars: inout [Character], _ visited: inout [String : Bool], _ res: inout Set<String>) {
        let m = board.count
        let n = board[0].count
        
        if chars.count > m + n {
            return
        }
        
        let string = chars2String(chars)
//        print("--- chars: \(chars), string: \(string) --- \n")
        if wordSet.contains(string) {
            res.insert(string)
        }
        
        if let hasVisited = visited[String(format: "%i-%i", row, col)], hasVisited {
            return
        }
        
        if row >= m || col >= n || row < 0 || col < 0 {
            return
        }
        
        chars.append(board[row][col])
        visited[String(format: "%i-%i", row, col)] = true
        dfsFindWords(board, wordSet, row + 1, col, &chars, &visited, &res)
        dfsFindWords(board, wordSet, row, col + 1, &chars, &visited, &res)
        dfsFindWords(board, wordSet, row - 1, col, &chars, &visited, &res)
        dfsFindWords(board, wordSet, row, col - 1, &chars, &visited, &res)
        chars.remove(at: chars.count - 1)
        visited[String(format: "%i-%i", row, col)] = false
    }
    
    private func chars2String(_ chars: [Character]) -> String {
        var string = ""
        var i = 0
        while i < chars.count {
            string += String(chars[i])
            i += 1
        }
        return string
    }
    
    /**
     844. 比较含退格的字符串
     
     给定 S 和 T 两个字符串，当它们分别被输入到空白的文本编辑器后，判断二者是否相等，并返回结果。 # 代表退格字符。

     注意：如果对空文本输入退格字符，文本继续为空。

      

     示例 1：

     输入：S = "ab#c", T = "ad#c"
     输出：true
     解释：S 和 T 都会变成 “ac”。
     示例 2：

     输入：S = "ab##", T = "c#d#"
     输出：true
     解释：S 和 T 都会变成 “”。
     示例 3：

     输入：S = "a##c", T = "#a#c"
     输出：true
     解释：S 和 T 都会变成 “c”。
     示例 4：

     输入：S = "a#c", T = "b"
     输出：false
     解释：S 会变成 “c”，但 T 仍然是 “b”。
      

     提示：

     1 <= S.length <= 200
     1 <= T.length <= 200
     S 和 T 只含有小写字母以及字符 '#'。
      

     进阶：

     你可以用 O(N) 的时间复杂度和 O(1) 的空间复杂度解决该问题吗？


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/backspace-string-compare
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func backspaceCompare(_ S: String, _ T: String) -> Bool {
        // 双指针法，从后往前，比较两个字符串中的字符，碰到 # 时，则指针向前移动两位
        let sArr = Array(S)
        let tArr = Array(T)
        var i = sArr.count - 1
        var j = tArr.count - 1
        var skipS = 0
        var skipT = 0
        while i >= 0 || j >= 0 {
            while i >= 0 {
                if "#" == sArr[i] {
                    skipS += 1
                    i -= 1
                } else if skipS > 0 {
                    skipS -= 1
                    i -= 1
                } else {
                    break
                }
            }
            
            while j >= 0 {
                if "#" == tArr[j] {
                    skipT += 1
                    j -= 1
                } else if skipT > 0 {
                    skipT -= 1
                    j -= 1
                } else {
                    break
                }
            }
            
            if i >= 0 && j >= 0 {
                if sArr[i] != tArr[j] {
                    return false
                }
            } else {
                if i >= 0 || j >= 0 {
                    return false
                }
            }

            i -= 1
            j -= 1
        }
        
        return true
        
        // 使用栈的解法
//        let sArr = Array(S)
//        let tArr = Array(T)
//        var sStack = [Character]()
//        var tStack = [Character]()
//        for char in sArr {
//            if char == "#" {
//                if !sStack.isEmpty {
//                    sStack.removeLast()
//                }
//            } else {
//                sStack.append(char)
//            }
//        }
//        for char in tArr {
//            if char == "#" {
//                if !tStack.isEmpty {
//                    tStack.removeLast()
//                }
//            } else {
//                tStack.append(char)
//            }
//        }
//        return sStack == tStack
    }
    
    /**
     76. 最小覆盖子串
     
     给你一个字符串 S、一个字符串 T 。请你设计一种算法，可以在 O(n) 的时间复杂度内，从字符串 S 里面找出：包含 T 所有字符的最小子串。

      

     示例：

     输入：S = "ADOBECODEBANC", T = "ABC"
     输出："BANC"
      

     提示：

     如果 S 中不存这样的子串，则返回空字符串 ""。
     如果 S 中存在这样的子串，我们保证它是唯一的答案。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/minimum-window-substring
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minWindow(_ s: String, _ t: String) -> String {
        let sArr = Array(s)
        let tArr = Array(t)
        var need = [Character : Int]()
        for char in tArr {
            if let count = need[char] {
                need[char] = count + 1
            } else {
                need[char] = 1
            }
        }
        
        var window = [Character : Int]()
        var left = 0, right = 0, valid = 0
        var start = 0, len = Int.max
        while right < sArr.count {
            let rightChar = sArr[right]
            right += 1
            
            if let _ = need[rightChar] {
                window[rightChar] = nil != window[rightChar] ? window[rightChar]! + 1 : 1
                if window[rightChar] == need[rightChar] {
                    valid += 1
                }
            }            
            
            while valid == need.count {
                if right - left < len {
                    len = right - left
                    start = left
                }
                
                let leftChar = sArr[left]
                left += 1
                
                if let _ = need[leftChar] {
                    if window[leftChar] == need[leftChar] {
                        valid -= 1
                    }
                    
                    if let count = window[leftChar] {
                        window[leftChar] = count - 1
                    }
                }
            }
        }
        
        if start < sArr.count && start + len <= sArr.count {
            return String(sArr[start ..< start + len])
        } else {
            return ""
        }
    }
    
    /**
     312. 戳气球
     
     有 n 个气球，编号为0 到 n-1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

     现在要求你戳破所有的气球。如果你戳破气球 i ，就可以获得 nums[left] * nums[i] * nums[right] 个硬币。 这里的 left 和 right 代表和 i 相邻的两个气球的序号。注意当你戳破了气球 i 后，气球 left 和气球 right 就变成了相邻的气球。

     求所能获得硬币的最大数量。

     说明:

     你可以假设 nums[-1] = nums[n] = 1，但注意它们不是真实存在的所以并不能被戳破。
     0 ≤ n ≤ 500, 0 ≤ nums[i] ≤ 100
     示例:

     输入: [3,1,5,8]
     输出: 167
     解释: nums = [3,1,5,8] --> [3,5,8] -->   [3,8]   -->  [8]  --> []
          coins =  3*1*5      +  3*5*8    +  1*3*8      + 1*8*1   = 167

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/burst-balloons
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxCoins(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 0 else {
            return 0
        }
        
        var val = [Int](repeating: 1, count: n + 2)
        for i in 0 ..< n {
            val[i + 1] = nums[i]
        }
        
        // dp[i][j] 表示填满开区间 (i, j) 能得到的最多硬币数
        var dp = [[Int]]()
        for _ in 0 ..< n + 2 {
            dp.append([Int](repeating: 0, count: n + 2))
        }
        
        for i in (0 ... n - 1).reversed() {
            for j in i + 2 ... n + 1 {
                for k in i + 1 ..< j {
                    var sum = val[i] * val[k] * val[j]
                    sum += (dp[i][k] + dp[k][j])
                    dp[i][j] = max(dp[i][j], sum)
                }
            }
        }
        
        return dp[0][n + 1]
    }
    
    /**
     200. 岛屿数量
     
     给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

     岛屿总是被水包围，并且每座岛屿只能由水平方向和/或竖直方向上相邻的陆地连接形成。

     此外，你可以假设该网格的四条边均被水包围。

      

     示例 1：

     输入：grid = [
       ["1","1","1","1","0"],
       ["1","1","0","1","0"],
       ["1","1","0","0","0"],
       ["0","0","0","0","0"]
     ]
     输出：1
     示例 2：

     输入：grid = [
       ["1","1","0","0","0"],
       ["1","1","0","0","0"],
       ["0","0","1","0","0"],
       ["0","0","0","1","1"]
     ]
     输出：3
      

     提示：

     m == grid.length
     n == grid[i].length
     1 <= m, n <= 300
     grid[i][j] 的值为 '0' 或 '1'

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/number-of-islands
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numIslands(_ grid: [[Character]]) -> Int {
        let m = grid.count
        guard m > 0 else {
            return 0
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return 0
        }
        
        var res = 0
        var tempGrid = grid
        for i in 0 ..< m {
            for j in 0 ..< n {
                if "1" == tempGrid[i][j] {
                    res += 1
                    dfsMarkIslands(&tempGrid, i, j, m, n)
                }
            }
        }
        
        return res
    }
    
    private func dfsMarkIslands(_ grid: inout [[Character]], _ row: Int, _ col: Int, _ m: Int, _ n: Int) {
        if row >= m || row < 0 || col >= n || col < 0 || "0" == grid[row][col] {
            return
        }
        
        grid[row][col] = "0"
        // 左 - 上 - 右 - 下
        let rowOffset = [-1, 0, 1, 0]
        let colOffset = [0, 1, 0, -1]
        for i in 0 ..< 4 {
            dfsMarkIslands(&grid, row + rowOffset[i], col + colOffset[i], m, n)
        }
    }
    
    /**
     36. 有效的数独
     
     判断一个 9x9 的数独是否有效。只需要根据以下规则，验证已经填入的数字是否有效即可。

     数字 1-9 在每一行只能出现一次。
     数字 1-9 在每一列只能出现一次。
     数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。


     上图是一个部分填充的有效的数独。

     数独部分空格内已填入了数字，空白格用 '.' 表示。

     示例 1:

     输入:
     [
       ["5","3",".",".","7",".",".",".","."],
       ["6",".",".","1","9","5",".",".","."],
       [".","9","8",".",".",".",".","6","."],
       ["8",".",".",".","6",".",".",".","3"],
       ["4",".",".","8",".","3",".",".","1"],
       ["7",".",".",".","2",".",".",".","6"],
       [".","6",".",".",".",".","2","8","."],
       [".",".",".","4","1","9",".",".","5"],
       [".",".",".",".","8",".",".","7","9"]
     ]
     输出: true
     示例 2:

     输入:
     [
       ["8","3",".",".","7",".",".",".","."],
       ["6",".",".","1","9","5",".",".","."],
       [".","9","8",".",".",".",".","6","."],
       ["8",".",".",".","6",".",".",".","3"],
       ["4",".",".","8",".","3",".",".","1"],
       ["7",".",".",".","2",".",".",".","6"],
       [".","6",".",".",".",".","2","8","."],
       [".",".",".","4","1","9",".",".","5"],
       [".",".",".",".","8",".",".","7","9"]
     ]
     输出: false
     解释: 除了第一行的第一个数字从 5 改为 8 以外，空格内其他数字均与 示例1 相同。
          但由于位于左上角的 3x3 宫内有两个 8 存在, 因此这个数独是无效的。
     说明:

     一个有效的数独（部分已被填充）不一定是可解的。
     只需要根据以上规则，验证已经填入的数字是否有效即可。
     给定数独序列只包含数字 1-9 和字符 '.' 。
     给定数独永远是 9x9 形式的。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-sudoku
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isValidSudoku(_ board: [[Character]]) -> Bool {
        guard 9 == board.count && 9 == board[0].count else {
            return false
        }
        
        var rows = [[Character : Int]]()
        var cols = [[Character : Int]]()
        var boxes = [[Character : Int]]()
        for _ in 0 ..< 9 {
            rows.append([Character : Int]())
            cols.append([Character : Int]())
            boxes.append([Character : Int]())
        }
        
        for i in 0 ..< 9 {
            for j in 0 ..< 9 {
                let char = board[i][j]
                if "." == char {
                    continue
                }
                
                if let _ = rows[i][char] {
                    return false
                }
                
                if let _ = cols[j][char] {
                    return false
                }
                
                let boxIndex = (i/3)*3 + j/3
                if let _ = boxes[boxIndex][char] {
                    return false
                }
                
                rows[i][char] = 1
                cols[j][char] = 1
                boxes[boxIndex][char] = 1
            }
        }
        
        return true
    }
    
    /**
     1091. 二进制矩阵中的最短路径
     
     在一个 N × N 的方形网格中，每个单元格有两种状态：空（0）或者阻塞（1）。

     一条从左上角到右下角、长度为 k 的畅通路径，由满足下述条件的单元格 C_1, C_2, ..., C_k 组成：

     相邻单元格 C_i 和 C_{i+1} 在八个方向之一上连通（此时，C_i 和 C_{i+1} 不同且共享边或角）
     C_1 位于 (0, 0)（即，值为 grid[0][0]）
     C_k 位于 (N-1, N-1)（即，值为 grid[N-1][N-1]）
     如果 C_i 位于 (r, c)，则 grid[r][c] 为空（即，grid[r][c] == 0）
     返回这条从左上角到右下角的最短畅通路径的长度。如果不存在这样的路径，返回 -1 。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/shortest-path-in-binary-matrix
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func shortestPathBinaryMatrix(_ grid: [[Int]]) -> Int {
        // BFS，何时用 BFS，何时用 DP ？
        let m = grid.count
        guard m > 0 else {
            return -1
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return -1
        }
        
        guard 0 == grid[0][0] && 0 == grid[m - 1][n - 1] else {
            return -1
        }
        
        var queue = [(x: Int, y: Int)]()
        queue.append((0, 0))
        var visited = [[Bool]]()
        for _ in 0 ..< m {
            visited.append([Bool](repeating: false, count: n))
        }
        visited[0][0] = true
        
        var res = 0
        // 左 - 左上 - 上 - 右上 - 右 - 右下 - 下 - 左下
        let xOffset = [-1, -1, 0, 1, 1, 1, 0, -1]
        let yOffset = [0, -1, -1, -1, 0, 1, 1, 1]
        while !queue.isEmpty {
            let size = queue.count
            for _ in 0 ..< size {
                let loc = queue.removeFirst()
                if loc.x == m - 1 && loc.y == n - 1 {
                    return res + 1
                }
                
                for i in 0 ..< xOffset.count {
                    let newX = loc.x + xOffset[i]
                    let newY = loc.y + yOffset[i]
                    if newX < 0 || newX >= m || newY < 0 || newY >= n || visited[newX][newY] {
                        continue
                    }
                    
                    if 1 == grid[newX][newY] {
                        continue
                    }
                    
                    queue.append((newX, newY))
                    visited[newX][newY] = true
                }
            }
            res += 1
        }
        
        return -1
        
        // 动态规划，case 无法通过
//        let m = grid.count
//        guard m > 0 else {
//            return -1
//        }
//
//        let n = grid[0].count
//        guard n > 0 else {
//            return -1
//        }
//
//        guard 0 == grid[m - 1][n - 1] else {
//            return -1
//        }
//
//        var dp = [[Int]]()
//        for _ in 0 ..< m {
//            dp.append([Int](repeating: -1, count: n))
//        }
//        dp[m - 1][n - 1] = 1
//
//        for i in (0 ..< m).reversed() {
//            for j in (0 ..< n).reversed() {
//                if i == m - 1 && j == n - 1 {
//                    continue
//                }
//
//                if 0 == grid[i][j] {
//                    let right = j + 1 < n ? dp[i][j + 1] : -1
//                    let down = i + 1 < m ? dp[i + 1][j] : -1
//                    let rightdown = (i + 1 < m && j + 1 < n) ? dp[i + 1][j + 1] : -1
//                    dp[i][j] = minValue(right, down, rightdown) + 1
//                }
//            }
//        }
//
//        return dp[0][0]
    }
    
    private func minValue(_ x: Int, _ y: Int, _ z: Int) -> Int {
        let value = min(x < 0 ? Int.max : x, y < 0 ? Int.max : y, z < 0 ? Int.max : z)
        return Int.max == value ? -2 : value
    }
    
    /**
     143. 重排链表
     
     给定一个单链表 L：L0→L1→…→Ln-1→Ln ，
     将其重新排列后变为： L0→Ln→L1→Ln-1→L2→Ln-2→…

     你不能只是单纯的改变节点内部的值，而是需要实际的进行节点交换。

     示例 1:

     给定链表 1->2->3->4, 重新排列为 1->4->2->3.
     示例 2:

     给定链表 1->2->3->4->5, 重新排列为 1->5->2->4->3.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reorder-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reorderList(_ head: ListNode?) {
        if nil == head || nil == head?.next || nil == head?.next?.next {
            return
        }
        
        var arr: [ListNode] = [ListNode]()
        var node = head
        while nil != node {
            arr.append(node!)
            node = node!.next
        }
        
        var i = 0, j = arr.count - 1
        while i < j {
            arr[i].next = arr[j]
            i += 1
            if i == j {
                break
            }
            arr[j].next = arr[j]
            j -= 1
        }
        arr[i].next = nil
    }
    
    private func reverseList(_ head: ListNode?) -> ListNode? {
        if nil == head || nil == head?.next {
            return head
        }
        
        var pre: ListNode? = nil
        var current = head
        var next: ListNode?
        while nil != current {
            next = current!.next
            current?.next = pre
            pre = current
            current = next
        }
        return pre
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
    func solveNQueens(_ n: Int) -> [[String]] {
        var res = [[String]]()
        guard n > 0 else {
            return res
        }
        
        var board = [[Character]]()
        for _ in 0 ..< n {
            board.append([Character](repeating: ".", count: n))
        }
        
        solveNQueensBacktrack(&board, n, 0, &res)
        
        return res
    }
    
    private func solveNQueensBacktrack(_ board: inout [[Character]], _ n: Int, _ row: Int, _ res: inout [[String]]) {
        if row == n {
            res.append(convertBoard(board, n))
            return
        }
        
        for col in 0 ..< n {
            // 判断能否在位置 (row, col) 放置皇后
            if !isValidNQueen(board, n, row, col) {
                continue
            }
            
            board[row][col] = "Q"
            solveNQueensBacktrack(&board, n, row + 1, &res)
            // 回溯
            board[row][col] = "."
        }
    }
    
    private func convertBoard(_ board: [[Character]], _ n: Int) -> [String] {
        var strs = [String]()
        for i in 0 ..< n {
            var str = ""
            for j in 0 ..< n {
                str += String(board[i][j])
            }
            strs.append(str)
        }
        return strs
    }
    
    private func isValidNQueen(_ board: [[Character]], _ n: Int, _ row: Int, _ col: Int) -> Bool {
        // 列
        for i in 0 ..< n {
            if "Q" == board[i][col] {
                return false
            }
        }
        
        // 左上方
        var i = row - 1, j = col - 1
        while i >= 0 && j >= 0 {
            if "Q" == board[i][j] {
                return false
            }
            
            i -= 1
            j -= 1
        }
        
        // 右上方
        i = row - 1
        j = col + 1
        while i >= 0 && j < n {
            if "Q" == board[i][j] {
                return false
            }
            
            i -= 1
            j += 1
        }
        
        return true
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
        guard beginWord.count > 0 && beginWord.count == endWord.count else {
            return 0
        }
        
        var wordSet = Set(wordList)
        guard wordSet.contains(endWord) else {
            return 0
        }
        wordSet.remove(beginWord)
        
        var queue = [String]()
        queue.append(beginWord)
        var visited = Set<String>()
        var res = 1
        while !queue.isEmpty {
            let size = queue.count
            for _ in 0 ..< size {
                // 队列，先进先出
                let word = queue.removeFirst()
                if changeWordEveryLetter(word, endWord, &queue, &visited, wordSet) {
                    return res + 1
                }
            }
            res += 1
        }
        return 0
    }
    
    private func changeWordEveryLetter(_ current: String, _ endWord: String, _ queue: inout [String], _ visited: inout Set<String>, _ wordSet: Set<String>) -> Bool {
        var currentArr = Array(current)
        for i in 0 ..< currentArr.count {
            let currentChar = currentArr[i]
            for letter in lowerLetters {
                if currentChar == letter {
                    continue
                }
                
                currentArr[i] = letter
                var nextWord = ""
                for char in currentArr {
                    nextWord += String(char)
                }
                
                if wordSet.contains(nextWord) {
                    if nextWord == endWord {
                        return true
                    }
                    
                    if !visited.contains(nextWord) {
                        queue.append(nextWord)
                        visited.insert(nextWord)
                    }
                }
            }
            currentArr[i] = currentChar
        }
        return false
    }
    
    /**
     925. 长按键入
     
     你的朋友正在使用键盘输入他的名字 name。偶尔，在键入字符 c 时，按键可能会被长按，而字符可能被输入 1 次或多次。

     你将会检查键盘输入的字符 typed。如果它对应的可能是你的朋友的名字（其中一些字符可能被长按），那么就返回 True。

      

     示例 1：

     输入：name = "alex", typed = "aaleex"
     输出：true
     解释：'alex' 中的 'a' 和 'e' 被长按。
     示例 2：

     输入：name = "saeed", typed = "ssaaedd"
     输出：false
     解释：'e' 一定需要被键入两次，但在 typed 的输出中不是这样。
     示例 3：

     输入：name = "leelee", typed = "lleeelee"
     输出：true
     示例 4：

     输入：name = "laiden", typed = "laiden"
     输出：true
     解释：长按名字中的字符并不是必要的。
      

     提示：

     name.length <= 1000
     typed.length <= 1000
     name 和 typed 的字符都是小写字母。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/long-pressed-name
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isLongPressedName(_ name: String, _ typed: String) -> Bool {
        let nameArr = Array(name)
        let typedArr = Array(typed)
        var i = 0, j = 0, m = nameArr.count, n = typedArr.count
        while j < n {
            if i < m && nameArr[i] == typedArr[j] {
                i += 1
                j += 1
            } else if j > 0 && typedArr[j] == typedArr[j - 1] {
                j += 1
            } else {
                return false
            }
        }
        return i == m
    }
    
    /**
     763. 划分字母区间
     
     字符串 S 由小写字母组成。我们要把这个字符串划分为尽可能多的片段，同一个字母只会出现在其中的一个片段。返回一个表示每个字符串片段的长度的列表。

      

     示例 1：

     输入：S = "ababcbacadefegdehijhklij"
     输出：[9,7,8]
     解释：
     划分结果为 "ababcbaca", "defegde", "hijhklij"。
     每个字母最多出现在一个片段中。
     像 "ababcbacadefegde", "hijhklij" 的划分是错误的，因为划分的片段数较少。
      

     提示：

     S的长度在[1, 500]之间。
     S只包含小写字母 'a' 到 'z' 。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/partition-labels
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func partitionLabels(_ S: String) -> [Int] {
        let sChars = Array(S)
        guard sChars.count > 0 else {
            return [0]
        }
        
        let lettera: Character = "a"
        var dict = [Int](repeating: 0, count: 26)
        for i in 0 ..< sChars.count {
            let char = sChars[i]
            dict[Int(char.asciiValue! - lettera.asciiValue!)] = max(dict[Int(char.asciiValue! - lettera.asciiValue!)], i)
        }
        
        var res = [Int]()
        var start = 0, end = 0
        for i in 0 ..< sChars.count {
            let char = sChars[i]
            end = max(dict[Int(char.asciiValue! - lettera.asciiValue!)], end)
            if i == end {
                res.append(end - start + 1)
                start = i + 1
                end = start
            }
        }
        return res
    }
    
    /**
     53. 最大子序和
     
     给定一个整数数组 nums ，找到一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

     示例:

     输入: [-2,1,-3,4,-1,2,1,-5,4]
     输出: 6
     解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
     进阶:

     如果你已经实现复杂度为 O(n) 的解法，尝试使用更为精妙的分治法求解。


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximum-subarray
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func maxSubArray(_ nums: [Int]) -> Int {
        guard nums.count > 0 else {
            return 0
        }
        
        var res = Int.min
        var sum = 0
        for i in 0 ..< nums.count {
            if sum > 0 {
                sum += nums[i]
            } else {
                sum = nums[i]
            }
            res = max(res, sum)
        }
        return res
    }
    
    /**
     234. 回文链表
     
     请判断一个链表是否为回文链表。

     示例 1:

     输入: 1->2
     输出: false
     示例 2:

     输入: 1->2->2->1
     输出: true
     进阶：
     你能否用 O(n) 时间复杂度和 O(1) 空间复杂度解决此题？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/palindrome-linked-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isPalindrome(_ head: ListNode?) -> Bool {
        if nil == head || nil == head?.next {
            return true
        }
        
        var arr = [ListNode]()
        var node = head
        while nil != node {
            arr.append(node!)
            node = node?.next
        }
        
        for i in 0 ... arr.count/2 - 1 {
            if arr[i].val != arr[arr.count - i - 1].val {
                return false
            }
        }
        
        return true
    }
    
    func isPalindrome2(_ head: ListNode?) -> Bool {
        if nil == head || nil == head?.next {
            return true
        }
        
        // 空间复杂度为 O(1) 解法的思路：使用快慢指针找到中点，然后将中点后的部分翻转，再逐一比较两个链表
        var slow = head
        var fast = head
        var mid: ListNode? = nil
        while fast != nil {
            if nil == fast?.next || nil == fast?.next?.next {
                mid = slow?.next
                slow?.next = nil
                break
            }
            
            slow = slow?.next
            fast = fast?.next?.next
        }

        slow = head
        // 翻转 mid
        var pre: ListNode? = nil
        var current = mid
        var next: ListNode? = nil
        while current != nil {
            next = current!.next
            current?.next = pre
            pre = current
            current = next
        }
        
        while pre != nil {
            if pre!.val != slow!.val {
                return false
            }
            pre = pre?.next
            slow = slow?.next
        }
        
        return true
    }
    
    /**
     1024. 视频拼接
     
     你将会获得一系列视频片段，这些片段来自于一项持续时长为 T 秒的体育赛事。这些片段可能有所重叠，也可能长度不一。

     视频片段 clips[i] 都用区间进行表示：开始于 clips[i][0] 并于 clips[i][1] 结束。我们甚至可以对这些片段自由地再剪辑，例如片段 [0, 7] 可以剪切成 [0, 1] + [1, 3] + [3, 7] 三部分。

     我们需要将这些片段进行再剪辑，并将剪辑后的内容拼接成覆盖整个运动过程的片段（[0, T]）。返回所需片段的最小数目，如果无法完成该任务，则返回 -1 。

      

     示例 1：

     输入：clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]], T = 10
     输出：3
     解释：
     我们选中 [0,2], [8,10], [1,9] 这三个片段。
     然后，按下面的方案重制比赛片段：
     将 [1,9] 再剪辑为 [1,2] + [2,8] + [8,9] 。
     现在我们手上有 [0,2] + [2,8] + [8,10]，而这些涵盖了整场比赛 [0, 10]。
     示例 2：

     输入：clips = [[0,1],[1,2]], T = 5
     输出：-1
     解释：
     我们无法只用 [0,1] 和 [1,2] 覆盖 [0,5] 的整个过程。
     示例 3：

     输入：clips = [[0,1],[6,8],[0,2],[5,6],[0,4],[0,3],[6,7],[1,3],[4,7],[1,4],[2,5],[2,6],[3,4],[4,5],[5,7],[6,9]], T = 9
     输出：3
     解释：
     我们选取片段 [0,4], [4,7] 和 [6,9] 。
     示例 4：

     输入：clips = [[0,4],[2,8]], T = 5
     输出：2
     解释：
     注意，你可能录制超过比赛结束时间的视频。
      

     提示：

     1 <= clips.length <= 100
     0 <= clips[i][0] <= clips[i][1] <= 100
     0 <= T <= 100

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/video-stitching
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func videoStitching(_ clips: [[Int]], _ T: Int) -> Int {
        // BFS + 回溯
        guard clips.count > 0 else {
            return -1
        }
        
        var tempClips = [[Int]]()
        for clip in clips {
            if clip[1] >= T {
                tempClips.append(clip)
            }
        }
        
        guard tempClips.count > 0 else {
            return -1
        }
        
        var res = Int.max
        for tempClip in tempClips {
            if tempClip[0] <= 0 && tempClip[1] >= T {
                return 1
            }
            
            var queue = [[Int]]()
            queue.append(tempClip)
            var clipSet = Set(clips)
            clipSet.remove(tempClip)
            
            var count = 1
            var found = false
            while !queue.isEmpty {
                if found {
                    break
                }
                let size = queue.count
                for _ in 0 ..< size {
                    if found {
                        break
                    }
                    let clip = queue.removeFirst()
                    for i in 0 ..< clips.count {
                        let currClip = clips[i]
                        if currClip[0] <= 0 && currClip[1] >= clip[0] {
                            res = min(res, count + 1)
                            found = true
                            continue
                        }
                        
                        if currClip[1] >= clip[0] && clipSet.contains(currClip) {
                            queue.append(currClip)
                            clipSet.remove(currClip)
                        }
                    }
                }
                count += 1
            }
        }
        
        return Int.max == res ? -1 : res
    }
}

/**
 208. 实现 Trie (前缀树)
 
 实现一个 Trie (前缀树)，包含 insert, search, 和 startsWith 这三个操作。

 示例:

 Trie trie = new Trie();

 trie.insert("apple");
 trie.search("apple");   // 返回 true
 trie.search("app");     // 返回 false
 trie.startsWith("app"); // 返回 true
 trie.insert("app");
 trie.search("app");     // 返回 true
 说明:

 你可以假设所有的输入都是由小写字母 a-z 构成的。
 保证所有输入均为非空字符串。

 来源：力扣（LeetCode）
 链接：https://leetcode-cn.com/problems/implement-trie-prefix-tree
 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 */
class Trie {
    private let root: TrieNode

    /** Initialize your data structure here. */
    init() {
        self.root = TrieNode()
    }
    
    /** Inserts a word into the trie. */
    func insert(_ word: String) {
        var current = self.root
        for char in Array(word) {
            if let node = current.children[char] {
                current = node
            } else {
                let node = TrieNode()
                current.children[char] = node
                current = node
            }
        }
        current.word = word
    }
    
    /** Returns if the word is in the trie. */
    func search(_ word: String) -> Bool {
        var node: TrieNode? = self.root
        for char in Array(word) {
            node = node?.children[char]
            if let tempWord = node?.word {
                if tempWord == word {
                    return true
                }
            }
        }
        return false
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    func startsWith(_ prefix: String) -> Bool {
        var node: TrieNode? = self.root
        for char in Array(prefix) {
            node = node?.children[char]
        }
        return nil != node
    }
}
