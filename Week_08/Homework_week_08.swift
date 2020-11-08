//
//  Homework_week_08.swift
//  DataStructDemo
//
//  Created by noctis on 2020/11/2.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

class Homework_week_08 {
    /**
     349. 两个数组的交集
     
     给定两个数组，编写一个函数来计算它们的交集。

      

     示例 1：

     输入：nums1 = [1,2,2,1], nums2 = [2,2]
     输出：[2]
     示例 2：

     输入：nums1 = [4,9,5], nums2 = [9,4,9,8,4]
     输出：[9,4]
      

     说明：

     输出结果中的每个元素一定是唯一的。
     我们可以不考虑输出结果的顺序。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/intersection-of-two-arrays
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        // 使用系统 API
//        return Array(Set(nums1).intersection(Set(nums2)))
        
        // 不使用系统 API
        var dict = [Int: Int]()
        for num in nums1 {
            dict[num] = nil == dict[num] ? 1 : dict[num]! + 1
        }
        
        var res = [Int]()
        for num in nums2 {
            if let _ = dict[num] {
                res.append(num)
                dict.removeValue(forKey: num)
            }
        }
        
        return res
    }
    
    /**
     387. 字符串中的第一个唯一字符
     
     给定一个字符串，找到它的第一个不重复的字符，并返回它的索引。如果不存在，则返回 -1。

      

     示例：

     s = "leetcode"
     返回 0

     s = "loveleetcode"
     返回 2
      

     提示：你可以假定该字符串只包含小写字母。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/first-unique-character-in-a-string
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func firstUniqChar(_ s: String) -> Int {
        var dict = [Character: Int]()
        let chars = Array(s)
        for char in chars {
            dict[char] = nil == dict[char] ? 1 : dict[char]! + 1
        }

        for i in 0 ..< chars.count {
            if 1 == dict[chars[i]]! {
                return i
            }
        }

        return -1
    }
    
    /**
     191. 位1的个数
     
     编写一个函数，输入是一个无符号整数，返回其二进制表达式中数字位数为 ‘1’ 的个数（也被称为汉明重量）。

      

     示例 1：

     输入：00000000000000000000000000001011
     输出：3
     解释：输入的二进制串 00000000000000000000000000001011 中，共有三位为 '1'。
     示例 2：

     输入：00000000000000000000000010000000
     输出：1
     解释：输入的二进制串 00000000000000000000000010000000 中，共有一位为 '1'。
     示例 3：

     输入：11111111111111111111111111111101
     输出：31
     解释：输入的二进制串 11111111111111111111111111111101 中，共有 31 位为 '1'。
      

     提示：

     请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
     在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 3 中，输入表示有符号整数 -3。
      

     进阶:
     如果多次调用这个函数，你将如何优化你的算法？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/number-of-1-bits
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func hammingWeight(_ n: Int) -> Int {
        var count = 0
        var x = n
        while x != 0 {
            x = x & (x - 1)     // 将 x 最低位的 1 清零
            count += 1
        }
        return count
    }
    
    /**
     231. 2的幂
     
     给定一个整数，编写一个函数来判断它是否是 2 的幂次方。

     示例 1:

     输入: 1
     输出: true
     解释: 20 = 1
     示例 2:

     输入: 16
     输出: true
     解释: 24 = 16
     示例 3:

     输入: 218
     输出: false

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/power-of-two
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func isPowerOfTwo(_ n: Int) -> Bool {
        // 二进制中有且仅有一个 1，则为 2 的幂
        return n > 0 && 0 == n & (n - 1)
    }
    
    /**
     190. 颠倒二进制位
     
     颠倒给定的 32 位无符号整数的二进制位。

      

     示例 1：

     输入: 00000010100101000001111010011100
     输出: 00111001011110000010100101000000
     解释: 输入的二进制串 00000010100101000001111010011100 表示无符号整数 43261596，
          因此返回 964176192，其二进制表示形式为 00111001011110000010100101000000。
     示例 2：

     输入：11111111111111111111111111111101
     输出：10111111111111111111111111111111
     解释：输入的二进制串 11111111111111111111111111111101 表示无符号整数 4294967293，
          因此返回 3221225471 其二进制表示形式为 10111111111111111111111111111111 。
      

     提示：

     请注意，在某些语言（如 Java）中，没有无符号整数类型。在这种情况下，输入和输出都将被指定为有符号整数类型，并且不应影响您的实现，因为无论整数是有符号的还是无符号的，其内部的二进制表示形式都是相同的。
     在 Java 中，编译器使用二进制补码记法来表示有符号整数。因此，在上面的 示例 2 中，输入表示有符号整数 -3，输出表示有符号整数 -1073741825。
      

     进阶:
     如果多次调用这个函数，你将如何优化你的算法？

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-bits
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reverseBits(_ n: Int) -> Int {
        var res = 0
        var x = n
        var index = 31
        while index >= 0 {
            res += (x & 1) << index
            x = x >> 1
            index -= 1
        }
        return res
    }
    
    /**
     338. 比特位计数
     
     给定一个非负整数 num。对于 0 ≤ i ≤ num 范围中的每个数字 i ，计算其二进制数中的 1 的数目并将它们作为数组返回。

     示例 1:

     输入: 2
     输出: [0,1,1]
     示例 2:

     输入: 5
     输出: [0,1,1,2,1,2]
     进阶:

     给出时间复杂度为O(n*sizeof(integer))的解答非常容易。但你可以在线性时间O(n)内用一趟扫描做到吗？
     要求算法的空间复杂度为O(n)。
     你能进一步完善解法吗？要求在C++或任何其他语言中不使用任何内置函数（如 C++ 中的 __builtin_popcount）来执行此操作。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/counting-bits
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func countBits(_ num: Int) -> [Int] {
//        var res = [Int]()
//        for i in 0 ... num {
//            var x = i
//            var count = 0
//            while x != 0 {
//                x = x & (x - 1)
//                count += 1
//            }
//            res.append(count)
//        }
//        return res
        
        // 动态规划 P(x)=P(x/2)+(xmod2)
//        var res = [Int](repeating: 0, count: num + 1)
//        for i in 1 ... num {
//            res[i] = res[i>>1] + (i&1)
//        }
//        return res
        
        // 动态规划 P(x)=P(x&(x−1))+1
        var res = [Int](repeating: 0, count: num + 1)
        for i in 1 ... num {
            res[i] = res[i&(i-1)] + 1
        }
        return res
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
        var result = [[Int]]()
        dfsSolveNQueens(n, [], [], [], &result)
        var res = [[String]]()
        for nums in result {
            var strs = [String]()
            for num in nums {
                var str = ""
                for i in 0 ..< n {
                    if i == num {
                        str += "Q"
                    } else {
                        str += "."
                    }
                }
                strs.append(str)
            }
            res.append(strs)
        }
        return res
    }
    
    private func dfsSolveNQueens(_ n: Int, _ queens: [Int], _ xyDif: [Int], _ xySum: [Int], _ result: inout [[Int]]) {
        let p = queens.count
        if p == n {
            result.append(queens)
            return
        }
        
        for q in 0 ..< n {
            if !queens.contains(q) && !xyDif.contains(p - q) && !xySum.contains(p + q) {
                dfsSolveNQueens(n, queens + [q], xyDif + [p - q], xySum + [p + q], &result)
            }
        }
    }
    
    /**
     941. 有效的山脉数组
     
     给定一个整数数组 A，如果它是有效的山脉数组就返回 true，否则返回 false。

     让我们回顾一下，如果 A 满足下述条件，那么它是一个山脉数组：

     A.length >= 3
     在 0 < i < A.length - 1 条件下，存在 i 使得：
     A[0] < A[1] < ... A[i-1] < A[i]
     A[i] > A[i+1] > ... > A[A.length - 1]
      



      

     示例 1：

     输入：[2,1]
     输出：false
     示例 2：

     输入：[3,5,5]
     输出：false
     示例 3：

     输入：[0,3,2,1]
     输出：true
      

     提示：

     0 <= A.length <= 10000
     0 <= A[i] <= 10000

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/valid-mountain-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func validMountainArray(_ A: [Int]) -> Bool {
//        guard A.count > 2 else {
//            return false
//        }
//
//        var maxNum = A[0]
//        var maxIndex = 0
//        for i in 1 ..< A.count {
//            if A[i] > maxNum {
//                maxNum = A[i]
//                maxIndex = i
//            }
//        }
//
//        if 0 == maxIndex || A.count - 1 == maxIndex {
//            return false
//        }
//
//        for i in 0 ..< maxIndex {
//            if A[i] >= A[i + 1] {
//                return false
//            }
//        }
//
//        for i in maxIndex ..< A.count - 1 {
//            if A[i] <= A[i + 1] {
//                return false
//            }
//        }
//
//        return true
        
        guard A.count > 2 else {
            return false
        }
        
        var longest = 0
        var start = -1
        for i in 1 ..< A.count {
            if A[i] > A[i - 1] {
                if 1 == i || A[i - 2] >= A[i - 1] {
                    start = i - 1
                }
            } else if A[i] < A[i - 1] {
                if -1 != start {
                    longest = max(longest, i - start + 1)
                }
            } else {
                start = -1
            }
        }
        return longest == A.count
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
            dp.append([Int](repeating: 0, count: n))
        }
        
        for i in 0 ..< m {
            dp[i][n - 1] = 1
        }
        
        for i in 0 ..< n {
            dp[m - 1][i] = 1
        }
        
        for i in (0 ..< m - 1).reversed() {
            for j in (0 ..< n - 1).reversed() {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1]
            }
        }
        
        return dp[0][0]
    }
    
    /**
     57. 插入区间
     
     给出一个无重叠的 ，按照区间起始端点排序的区间列表。

     在列表中插入一个新的区间，你需要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

      

     示例 1：

     输入：intervals = [[1,3],[6,9]], newInterval = [2,5]
     输出：[[1,5],[6,9]]
     示例 2：

     输入：intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
     输出：[[1,2],[3,10],[12,16]]
     解释：这是因为新的区间 [4,8] 与 [3,5],[6,7],[8,10] 重叠。
      

     注意：输入类型已在 2019 年 4 月 15 日更改。请重置为默认代码定义以获取新的方法签名。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/insert-interval
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func insert(_ intervals: [[Int]], _ newInterval: [Int]) -> [[Int]] {
        guard intervals.count > 0 && 2 == newInterval.count else {
            return 2 == newInterval.count ? [newInterval] : intervals
        }
        
        var start = newInterval[0], end = newInterval[1]
        var placed = false
        var res = [[Int]]()
        
        for interval in intervals {
            if interval[0] > end {
                if !placed {
                    res.append([start, end])
                    placed = true
                }
                res.append(interval)
            } else if interval[1] < start {
                res.append(interval)
            } else {
                start = min(start, interval[0])
                end = max(end, interval[1])
            }
        }
        
        if !placed {
            res.append([start, end])
        }
        
        return res
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
        guard k > 0 else {
            return s
        }
        
        var arr = Array(s)
        let count = arr.count
        var i = 0
        while i < count {
            var tempArr = Array(arr[i ..< (i + k > count ? count : i + k)])
            reverseCharArr(&tempArr)
            for j in 0 ..< tempArr.count {
                arr[i + j] = tempArr[j]
            }
            if i + k > count {
                break
            } else {
                i += 2 * k
            }
        }
        
        return arr.compactMap { "\($0)" }.joined()
    }
    
    private func reverseCharArr(_ arr: inout [Character]) {
        for i in 0 ..< arr.count/2 {
            let temp = arr[arr.count - 1 - i]
            arr[arr.count - 1 - i] = arr[i]
            arr[i] = temp
        }
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
        var dict = [Character: Int]()
        for char in Array(s) {
            dict[char] = nil == dict[char] ? 1 : dict[char]! + 1
        }
        
        for char in Array(t) {
            if let count = dict[char] {
                dict[char] = count - 1
            } else {
                dict[char] = 1
            }
        }
        
        for value in dict.values {
            if 0 != value {
                return false
            }
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
        let wordSet = Set(wordList)
        guard wordSet.contains(endWord) else {
            return 0
        }
        
        var beginSet = Set<String>(), endSet = Set<String>(), visited = Set<String>()
        beginSet.insert(beginWord)
        endSet.insert(endWord)
        
        var len = 1
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
                    let wordChar = wordArr[i]
                    for letter in letters {
                        if wordChar == letter {
                            continue
                        }
                        
                        wordArr[i] = letter
                        // 使用 compactMap 效率较低
//                        let nextWord = wordArr.compactMap { "\($0)" }.joined()
                        var nextWord = ""
                        for j in 0 ..< wordArr.count {
                            nextWord.append(wordArr[j])
                        }
                        if endSet.contains(nextWord) {
                            return len + 1
                        }
                        
                        if wordSet.contains(nextWord) && !visited.contains(nextWord) {
                            tempSet.insert(nextWord)
                            visited.insert(nextWord)
                        }
                    }
                    wordArr[i] = wordChar
                }
            }
            
            len += 1
            beginSet = tempSet
        }
        
        return 0
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
        
        var len = 0
        var dp = [Int](repeating: 1, count: n)
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
     56. 合并区间
     
     给出一个区间的集合，请合并所有重叠的区间。

      

     示例 1:

     输入: intervals = [[1,3],[2,6],[8,10],[15,18]]
     输出: [[1,6],[8,10],[15,18]]
     解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
     示例 2:

     输入: intervals = [[1,4],[4,5]]
     输出: [[1,5]]
     解释: 区间 [1,4] 和 [4,5] 可被视为重叠区间。
     注意：输入类型已于2019年4月15日更改。 请重置默认代码定义以获取新方法签名。

      

     提示：

     intervals[i][0] <= intervals[i][1]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/merge-intervals
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func merge(_ intervals: [[Int]]) -> [[Int]] {
        guard intervals.count > 1 else {
            return intervals
        }
        
        let sortedIntervals = intervals.sorted { (a, b) -> Bool in
            return a[0] < b[0]
        }
        
        var res = [[Int]]()
        res.append(sortedIntervals[0])
        for i in 1 ..< sortedIntervals.count {
            var resInterval = res.last!
            let interval = sortedIntervals[i]
            if resInterval[1] >= interval[0] || resInterval[1] >= interval[1] {
                resInterval[1] = max(resInterval[1], interval[1])
                res[res.count - 1] = resInterval
            } else {
                res.append(interval)
            }
        }
        return res
    }
    
    func countRangeSum(_ nums: [Int], _ lower: Int, _ upper: Int) -> Int {
        let n = nums.count
        guard n > 0 else {
            return 0
        }
        
        var sum = 0
        // 前缀和数组，求所有满足 sums[j] - sums[i] ∈ [lower, upper] 的下标对 (i, j)
        var sums = [Int](repeating: 0, count: n + 1)
        for i in 0 ..< n {
            sum += nums[i]
            sums[i + 1] = sum
        }
        return countRangeSumHelper(&sums, 0, n, lower, upper)
    }
    
    /**
     327. 区间和的个数
     
     给定一个整数数组 nums，返回区间和在 [lower, upper] 之间的个数，包含 lower 和 upper。
     区间和 S(i, j) 表示在 nums 中，位置从 i 到 j 的元素之和，包含 i 和 j (i ≤ j)。

     说明:
     最直观的算法复杂度是 O(n2) ，请在此基础上优化你的算法。

     示例:

     输入: nums = [-2,5,-1], lower = -2, upper = 2,
     输出: 3
     解释: 3个区间分别是: [0,0], [2,2], [0,2]，它们表示的和分别为: -2, -1, 2。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/count-of-range-sum
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    private func countRangeSumHelper(_ nums: inout [Int], _ left: Int, _ right: Int, _ lower: Int, _ upper: Int) -> Int {
        if left == right {
            return 0
        } else {
            let mid = left + (right - left) >> 1
            // 左半部分下标对数量
            let n1 = countRangeSumHelper(&nums, left, mid, lower, upper)
            // 右半部分下标对数量
            let n2 = countRangeSumHelper(&nums, mid + 1, right, lower, upper)
            var res = n1 + n2
            
            // 统计两个已升序排列的数组中的下标对数量
            var i = left
            var l = mid + 1
            var r = mid + 1
            while i <= mid {
                while l <= right && nums[l] - nums[i] < lower {
                    l += 1
                }
                
                while r <= right && nums[r] - nums[i] <= upper {
                    r += 1
                }
                
                res += (r - l)
                i += 1
            }
            
            // 合并两个排序的数组
            var sorted = [Int]()
            var p1 = left, p2 = mid + 1
            while p1 <= mid && p2 <= right {
                if nums[p2] > nums[p1] {
                    sorted.append(nums[p1])
                    p1 += 1
                } else {
                    sorted.append(nums[p2])
                    p2 += 1
                }
            }
            
            while p1 <= mid {
                sorted.append(nums[p1])
                p1 += 1
            }
            
            while p2 <= right {
                sorted.append(nums[p2])
                p2 += 1
            }
            
            for j in 0 ..< sorted.count {
                nums[left + j] = sorted[j]
            }
            
            return res
        }
    }
    
    /**
     493. 翻转对
     
     给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。

     你需要返回给定数组中的重要翻转对的数量。

     示例 1:

     输入: [1,3,2,3,1]
     输出: 2
     示例 2:

     输入: [2,4,3,5,1]
     输出: 3
     注意:

     给定数组的长度不会超过50000。
     输入数组中的所有数字都在32位整数的表示范围内。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/reverse-pairs
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func reversePairs(_ nums: [Int]) -> Int {
        let n = nums.count
        guard n > 1 else {
            return 0
        }
        
        var tempNums = nums
        return reversePairsHelper(&tempNums, 0, n - 1)
    }
    
    private func reversePairsHelper(_ nums: inout [Int], _ left: Int, _ right: Int) -> Int {
        if left == right {
            return 0
        } else {
            let mid = left + (right - left) >> 1
            let n1 = reversePairsHelper(&nums, left, mid)
            let n2 = reversePairsHelper(&nums, mid + 1, right)
            var res = n1 + n2
            
            // 统计两个排序数组中的翻转对
            var i = left, j = mid + 1
            while i <= mid {
                while j <= right && nums[i] > 2 * nums[j] {
                    j += 1
                }
                res += (j - mid - 1)
                i += 1
            }
            
            // 合并两个排序数组
            var p1 = left, p2 = mid + 1
            var sorted = [Int]()
            while p1 <= mid && p2 <= right {
                if nums[p1] < nums[p2] {
                    sorted.append(nums[p1])
                    p1 += 1
                } else {
                    sorted.append(nums[p2])
                    p2 += 1
                }
            }
            
            while p1 <= mid {
                sorted.append(nums[p1])
                p1 += 1
            }
            
            while p2 <= right {
                sorted.append(nums[p2])
                p2 += 1
            }
            
            for k in 0 ..< sorted.count {
                nums[left + k] = sorted[k]
            }
            
            return res
        }
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
        let chars = Array(s)
        var low = 0, high = chars.count - 1
        while low < high {
            if chars[low] == chars[high] {
                low += 1
                high -= 1
            } else {
                var flag1 = true, flag2 = true
                var i = low, j = high - 1   // 删掉右边字符
                while i < j {
                    if chars[i] != chars[j] {
                        flag1 = false
                        break
                    }
                    i += 1
                    j -= 1
                }
                
                // 删掉左边字符
                i = low + 1
                j = high
                while i < j {
                    if chars[i] != chars[j] {
                        flag2 = false
                        break
                    }
                    i += 1
                    j -= 1
                }
                
                return flag1 || flag2
            }
        }
        return true
    }
    
    private func validPalindromeChars(_ chars: [Character]) -> Bool {
        guard chars.count > 1 else {
            return true
        }
        
        for i in 0 ... (0 == chars.count%2 ? chars.count >> 1 - 1 : chars.count >> 1) {
            if chars[i] != chars[chars.count - 1 - i] {
                return false
            }
        }
        
        return true
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
    func maxProfit(_ prices: [Int]) -> Int {
        let n = prices.count
        guard n > 1 else {
            return 0
        }
        
        // dp[i][0] - 第 i 天不持有股票的最大利润，dp[i][1] - 第 i 天持有股票的最大利润
        var dp = [[Int]]()
        for _ in 0 ..< n {
            dp.append([Int](repeating: 0, count: 2))
        }
        
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        
        for i in 1 ..< n {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
        }
        
        return dp[n - 1][0]
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
        
        // dp[i] 表示 sArr[0 ..< i] 的最长有效括号长度
        var dp = [Int](repeating: 0, count: sArr.count)
        var res = 0
        for i in 1 ..< sArr.count {
            if ")" == sArr[i] {
                if "(" == sArr[i - 1] {
                    dp[i] = i - 2 >= 0 ? dp[i - 2] + 2 : 2
                } else if i - dp[i - 1] > 0 && "(" == sArr[i - dp[i - 1] - 1] {
                    dp[i] = i - dp[i - 1] > 1 ? dp[i - 1] + dp[i - dp[i - 1] - 2] + 2 : dp[i - 1] + 2
                }
            }
            res = max(res, dp[i])
        }
        return res
    }
}

class Homework_week_08_Sort {
    /**
     冒泡排序
     
     1 比较相邻的元素。如果第一个比第二个大，就交换它们两个；
     2 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对，这样在最后的元素应该会是最大的数；
     3 针对所有的元素重复以上的步骤，除了最后一个；
     4 重复步骤1~3，直到排序完成
     */
    func bubbleSort(_ arr: inout [Int]) {
        let n = arr.count
        for i in 0 ..< n - 1 {
            for j in 0 ..< n - 1 - i {
                if arr[j] > arr[j + 1] {
                    let temp = arr[j]
                    arr[j] = arr[j + 1]
                    arr[j + 1] = temp
                }
            }
        }
    }
    
    /**
     选择排序
     
     首先在未排序序列中找到最小（大）元素，存放到排序序列的起始位置，然后，再从剩余未排序元素中继续寻找最小（大）元素，然后放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

     1 初始状态：无序区为R[1..n]，有序区为空；
     2 第i趟排序(i=1,2,3…n-1)开始时，当前有序区和无序区分别为R[1..i-1]和R(i..n）。该趟排序从当前无序区中-选出关键字最小的记录 R[k]，将它与无序区的第1个记录R交换，使R[1..i]和R[i+1..n)分别变为记录个数增加1个的新有序区和记录个数减少1个的新无序区；
     3 n-1趟结束，数组有序化了
     */
    func selectSort(_ arr: inout [Int]) {
        let n = arr.count
        for i in 0 ..< n - 1 {
            var minIndex = i
            for j in i + 1 ..< n {
                if arr[j] < arr[minIndex] {
                    minIndex = j
                }
            }
            
            if i == minIndex {
                continue
            }
            
            let temp = arr[i]
            arr[i] = arr[minIndex]
            arr[minIndex] = temp
        }
    }
    
    /**
     插入排序
     
     通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入
     
     1 从第一个元素开始，该元素可以认为已经被排序；
     2 取出下一个元素，在已经排序的元素序列中从后向前扫描；
     3 如果该元素（已排序）大于新元素，将该元素移到下一位置；
     4 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置；
     5 将新元素插入到该位置后；
     6 重复步骤2~5
     */
    func insertSort(_ arr: inout [Int]) {
        let n = arr.count
        for i in 1 ..< n {
            let curr = arr[i]
            var index = i - 1
            while index >= 0 && arr[index] > curr { // 只挪比 curr 大的
                arr[index + 1] = arr[index]
                index -= 1
            }
            arr[index + 1] = curr
        }
    }
    
    /**
     希尔排序（Shell Sort）
     
     先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序，具体算法描述：

     1 选择一个增量序列t1，t2，…，tk，其中ti>tj，tk=1；
     2 按增量序列个数k，对序列进行k 趟排序；
     3 每趟排序，根据对应的增量ti，将待排序列分割成若干长度为m 的子序列，分别对各子表进行直接插入排序。仅增量因子为1 时，整个序列作为一个表来处理，表长度即为整个序列的长度
     */
    func shellSort(_ arr: inout [Int]) {
        let n = arr.count
        var gap = n/2
        while gap > 0 {
            for i in gap ..< n {
                var j = i
                let current = arr[i]
                while j - gap >= 0 && current < arr[j - gap] {
                    arr[j] = arr[j - gap]
                    j -= gap
                }
                arr[j] = current
            }
            gap = gap/2
        }
    }
    
    /**
     归并排序 - 树的后序遍历
     */
    func mergeSort(_ arr: [Int]) -> [Int] {
        if arr.count < 2 {
            return arr
        }
        
        let mid = arr.count >> 1
        let left = Array(arr[0 ..< mid])
        let right = Array(arr[mid ..< arr.count])
        let mergedLeft = mergeSort(left)
        let mergedRight = mergeSort(right)
        return merge(mergedLeft, mergedRight)
    }
    
    private func merge(_ arrA: [Int], _ arrB: [Int]) -> [Int] {
        var res = [Int]()
        let m = arrA.count, n = arrB.count
        var i = 0, j = 0
        while i < m && j < n {
            if arrA[i] > arrB[j] {
                res.append(arrB[j])
                j += 1
            } else {
                res.append(arrA[i])
                i += 1
            }
        }
        
        while i < m {
            res.append(arrA[i])
            i += 1
        }
        
        while j < n {
            res.append(arrB[j])
            j += 1
        }
        
        return res
    }
    
    /**
     快速排序 - 树的前序遍历
     */
    func quickSort(_ arr: inout [Int]) {
        quickSort(&arr, 0, arr.count - 1)
    }
    
    private func quickSort(_ arr: inout [Int], _ start: Int, _ end: Int) {
        guard arr.count > 1 && end > start else {
            return
        }
        
        let index = partition(&arr, start, end)
        if index > start {
            quickSort(&arr, start, index - 1)
        }
        if index < end {
            quickSort(&arr, index + 1, end)
        }
    }
    
    private func partition(_ arr: inout [Int], _ start: Int, _ end: Int) -> Int {
        let pivot = end     // 标杆位置
        var counter = start // 小于 pivot 的元素的个数
        for i in start ..< end {
            if arr[i] < arr[pivot] {    // 所有小于 pivot 的元素交换到 pivot 前面
                swap(&arr, counter, i)
                counter += 1
            }
        }
        
        swap(&arr, counter, pivot)
        return counter
    }
    
    private func swap(_ arr: inout [Int], _ start: Int, _ end: Int) {
        let temp = arr[start]
        arr[start] = arr[end]
        arr[end] = temp
    }
}

/**
 146. LRU缓存机制
 
 运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。它应该支持以下操作： 获取数据 get 和 写入数据 put 。

 获取数据 get(key) - 如果关键字 (key) 存在于缓存中，则获取关键字的值（总是正数），否则返回 -1。
 写入数据 put(key, value) - 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字/值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。

  

 进阶:

 你是否可以在 O(1) 时间复杂度内完成这两种操作？

  

 示例:

 LRUCache cache = new LRUCache( 2 /* 缓存容量 */ );

 cache.put(1, 1);
 cache.put(2, 2);
 cache.get(1);       // 返回  1
 cache.put(3, 3);    // 该操作会使得关键字 2 作废
 cache.get(2);       // 返回 -1 (未找到)
 cache.put(4, 4);    // 该操作会使得关键字 1 作废
 cache.get(1);       // 返回 -1 (未找到)
 cache.get(3);       // 返回  3
 cache.get(4);       // 返回  4

 来源：力扣（LeetCode）
 链接：https://leetcode-cn.com/problems/lru-cache
 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 */
class LinkedHashNode {
    var key: Int
    var value: Int
    var pre: LinkedHashNode?
    var next: LinkedHashNode?
    
    init(_ key: Int, _ value: Int) {
        self.key = key
        self.value = value
    }
}

class LinkedHashList {
    private var head: LinkedHashNode
    private var tail: LinkedHashNode
    private var size: Int
    
    init() {
        self.head = LinkedHashNode(0, 0)
        self.tail = LinkedHashNode(0, 0)
        self.head.next = tail
        self.tail.pre = head
        self.size = 0
    }
    
    func addLast(_ node: LinkedHashNode) {
        tail.pre?.next = node
        node.pre = tail.pre
        tail.pre = node
        node.next = tail
        size += 1
    }
    
    func remove(_ node: LinkedHashNode) {
        node.pre?.next = node.next
        node.next?.pre = node.pre
        size -= 1
    }
    
    func removeFirst() -> LinkedHashNode? {
        if head.next?.next == nil {
            return nil
        }
        
        let node = head.next
        remove(node!)
        return node
    }
    
    func capacity() -> Int {
        return size
    }
}

class LRUCache {
    private var dict: [Int: LinkedHashNode]
    private var linkedHashList: LinkedHashList
    private var capacity: Int

    init(_ capacity: Int) {
        self.dict = [Int: LinkedHashNode]()
        self.linkedHashList = LinkedHashList()
        self.capacity = capacity
    }
    
    func get(_ key: Int) -> Int {
        if let node = dict[key] {
            makeRecently(node.key)
            return node.value
        }
        
        return -1
    }
    
    func put(_ key: Int, _ value: Int) {
        if dict.keys.contains(key) {
            deleteKey(key)
            addRecently(key, value)
            return
        }
        
        if capacity == linkedHashList.capacity() {
            removeLeastRecently()
        }
        
        addRecently(key, value)
    }
    
    private func makeRecently(_ key: Int) {
        if let node = dict[key] {
            linkedHashList.remove(node)
            linkedHashList.addLast(node)
        }
    }
    
    private func addRecently(_ key: Int, _ value: Int) {
        let node = LinkedHashNode(key, value)
        linkedHashList.addLast(node)
        dict[key] = node
    }
    
    private func deleteKey(_ key: Int) {
        if let node = dict[key] {
            linkedHashList.remove(node)
            dict.removeValue(forKey: key)
        }
    }
    
    private func removeLeastRecently() {
        if let node = linkedHashList.removeFirst() {
            dict.removeValue(forKey: node.key)
        }
    }
}
