//
//  Homework_week_06.swift
//  DataStructDemo
//
//  Created by 刘练 on 2020/10/25.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

class Homework_week_07 {
    /**
     547. 朋友圈
     
     班上有 N 名学生。其中有些人是朋友，有些则不是。他们的友谊具有是传递性。如果已知 A 是 B 的朋友，B 是 C 的朋友，那么我们可以认为 A 也是 C 的朋友。所谓的朋友圈，是指所有朋友的集合。

     给定一个 N * N 的矩阵 M，表示班级中学生之间的朋友关系。如果M[i][j] = 1，表示已知第 i 个和 j 个学生互为朋友关系，否则为不知道。你必须输出所有学生中的已知的朋友圈总数。

      

     示例 1：

     输入：
     [[1,1,0],
      [1,1,0],
      [0,0,1]]
     输出：2
     解释：已知学生 0 和学生 1 互为朋友，他们在一个朋友圈。
     第2个学生自己在一个朋友圈。所以返回 2 。
     示例 2：

     输入：
     [[1,1,0],
      [1,1,1],
      [0,1,1]]
     输出：1
     解释：已知学生 0 和学生 1 互为朋友，学生 1 和学生 2 互为朋友，所以学生 0 和学生 2 也是朋友，所以他们三个在一个朋友圈，返回 1 。
      

     提示：

     1 <= N <= 200
     M[i][i] == 1
     M[i][j] == M[j][i]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/friend-circles
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findCircleNum(_ M: [[Int]]) -> Int {
        // DFS 解法
//        let n = M.count
//        guard n > 0 && M[0].count == n else {
//            return 0
//        }
//
//        var res = 0
//        var visited = [Bool](repeating: false, count: n)
//        for i in 0 ..< n {
//            if !visited[i] {
//                dfsMarkCircleNum(M, &visited, i)
//                res += 1
//            }
//        }
//        return res
        
        // 并查集解法
        let n = M.count
        guard n > 0 && M[0].count == n else {
            return 0
        }
        
        let uf = UnionFind(n)
        for i in 0 ..< n {
            for j in 0 ..< n {
                if 1 == M[i][j] {
                    uf.union(i, j)
                }
            }
        }
        return uf.getCount()
    }
    
    private func dfsMarkCircleNum(_ M: [[Int]], _ visited: inout [Bool], _ row: Int) {
        for col in 0 ..< M.count {
            if 1 == M[row][col] && !visited[col] {
                visited[col] = true
                dfsMarkCircleNum(M, &visited, col)
            }
        }
    }
    
    /**
     130. 被围绕的区域
     
     给定一个二维的矩阵，包含 'X' 和 'O'（字母 O）。

     找到所有被 'X' 围绕的区域，并将这些区域里所有的 'O' 用 'X' 填充。

     示例:

     X X X X
     X O O X
     X X O X
     X O X X
     运行你的函数后，矩阵变为：

     X X X X
     X X X X
     X X X X
     X O X X
     解释:

     被围绕的区间不会存在于边界上，换句话说，任何边界上的 'O' 都不会被填充为 'X'。 任何不在边界上，或不与边界上的 'O' 相连的 'O' 最终都会被填充为 'X'。如果两个元素在水平或垂直方向相邻，则称它们是“相连”的。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/surrounded-regions
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func solve(_ board: inout [[Character]]) {
        let m = board.count
        guard m > 0 else {
            return
        }
        
        let n = board[0].count
        guard n > 0 else {
            return
        }
        
        // 将四个边界中的 O 及与四个边界中的 O 相连的 O 修改为 #
        let edges = [(0, 0), (m - 1, 0), (0, 1), (n - 1, 1)]
        for edge in edges {
            let direction = edge.1
            let x = edge.0
            if 0 == direction {
                for i in 0 ..< n {
                    if board[x][i] == "O" {
                        dfsSolve(&board, m, n, x, i)
                    }
                }
            } else {
                for i in 0 ..< m {
                    if board[i][x] == "O" {
                        dfsSolve(&board, m, n, i, x)
                    }
                }
            }
        }
        
        for i in 0 ..< m {
            for j in 0 ..< n {
                if "#" == board[i][j] {
                    board[i][j] = "O"
                } else if "O" == board[i][j] {
                    board[i][j] = "X"
                }
            }
        }
    }
    
    private func dfsSolve(_ board: inout [[Character]], _ m: Int, _ n: Int, _ row: Int, _ col: Int) {
        if row < 0 || row >= m || col < 0 || col >= n || "#" == board[row][col] {
            return
        }
        
        if "O" == board[row][col] {
            board[row][col] = "#"
            dfsSolve(&board, m, n, row - 1, col)
            dfsSolve(&board, m, n, row + 1, col)
            dfsSolve(&board, m, n, row, col - 1)
            dfsSolve(&board, m, n, row, col + 1)
        }
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
        // 迭代解法
        var res = [Int]()
        var stack = [TreeNode]()
        if let node = root {
            stack.append(node)
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
        
        // 递归解法
//        var res = [Int]()
//        preorderHelper(root, &res)
//        return res
    }
    
    private func preorderHelper(_ root: TreeNode?, _ res: inout [Int]) {
        if nil == root {
            return
        }
        
        res.append(root!.val)
        preorderHelper(root?.left, &res)
        preorderHelper(root?.right, &res)
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
        // DFS 解法
//        let m = grid.count
//        guard m > 0 else {
//            return 0
//        }
//
//        let n = grid[0].count
//        guard n > 0 else {
//            return 0
//        }
//
//        var res = 0
//        var tempGrid = grid
//        for i in 0 ..< m {
//            for j in 0 ..< n {
//                if "1" == tempGrid[i][j] {
//                    res += 1
//                    dfsMarkIslands(&tempGrid, i, j)
//                }
//            }
//        }
//        return res
        
        // 并查集解法
        let m = grid.count
        guard m > 0 else {
            return 0
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return 0
        }
        
        let directions: [(x: Int, y: Int)] = [(1, 0), (0, 1)] // 右边和下边
        let uf = UnionFind(m * n + 1)
        let dummy = m * n
        for i in 0 ..< m {
            for j in 0 ..< n {
                // 水域全部并到 dummy
                if "0" == grid[i][j] {
                    uf.union(i * n + j, dummy)
                } else if "1" == grid[i][j] {
                    for direction in directions {
                        if i + direction.x < m && j + direction.y < n && "1" == grid[i + direction.x][j + direction.y] {
                            uf.union((i + direction.x) * n + j + direction.y, i * n + j)
                        }
                    }
                }
            }
        }
        return uf.getCount() - 1 // 减掉水域数量
    }
    
    private func dfsMarkIslands(_ grid: inout [[Character]], _ row: Int, _ col: Int) {
        let m = grid.count, n = grid[0].count
        if row < 0 || row >= m || col < 0 || col >= n || "0" == grid[row][col] {
            return
        }
        
        grid[row][col] = "0"
        let xOffset = [-1, 0, 1, 0]
        let yOffset = [0, -1, 0, 1]
        for i in 0 ..< 4 {
            dfsMarkIslands(&grid, row + xOffset[i], col + yOffset[i])
        }
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
        guard num >= 0 else {
            return false
        }
        
        var cur = 1.0
        while true {
            let pre = cur
            cur = (cur + Double(num)/cur)/2
            if fabs(pre - cur) < 0.000001 {
                break
            }
        }
        return fabs(cur - Double(Int(cur))) < 0.000001
    }
    
    /**
     1207. 独一无二的出现次数
     
     给你一个整数数组 arr，请你帮忙统计数组中每个数的出现次数。

     如果每个数的出现次数都是独一无二的，就返回 true；否则返回 false。

      

     示例 1：

     输入：arr = [1,2,2,1,1,3]
     输出：true
     解释：在该数组中，1 出现了 3 次，2 出现了 2 次，3 只出现了 1 次。没有两个数的出现次数相同。
     示例 2：

     输入：arr = [1,2]
     输出：false
     示例 3：

     输入：arr = [-3,0,1,-3,1,1,1,-3,10,0]
     输出：true

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/unique-number-of-occurrences
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func uniqueOccurrences(_ arr: [Int]) -> Bool {
        var dict = [Int: Int]()
        for num in arr {
            if let count = dict[num] {
                dict[num] = count + 1
            } else {
                dict[num] = 1
            }
        }
        
        var set = Set<Int>()
        for value in dict.values {
            if set.contains(value) {
                return false
            }
            
            set.insert(value)
        }
        
        return true
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
    func findWords(_ board: [[Character]], _ words: [String]) -> [String] {
        var res = [String]()
        guard words.count > 0 else {
            return res
        }
        
        let m = board.count
        guard m > 0 else {
            return res
        }
        
        let n = board[0].count
        guard n > 0 else {
            return res
        }
        
        let trie = Trie()
        for word in words {
            trie.insert(word)
        }
        
        var tempBoard = board
        var resSet = Set<String>()
        for i in 0 ..< m {
            for j in 0 ..< n {
                if trie.root.children.keys.contains(board[i][j]) {
                    dfsFindWords(trie.root, &tempBoard, i, j, &resSet)
                }
            }
        }
        
        res = Array(resSet)
        return res
    }
    
    private func dfsFindWords(_ root: TrieNode, _ board: inout [[Character]], _ row: Int, _ col: Int, _ res: inout Set<String>) {
        let letter = board[row][col]
        let currentNode = root.children[letter]
        if nil != currentNode && nil != currentNode!.word {
            res.insert(currentNode!.word!)
            currentNode!.word = nil
        }
        
        let xOffset = [-1, 0, 1, 0]
        let yOffset = [0, -1, 0, 1]
        board[row][col] = "#"
        for i in 0 ..< 4 {
            let newRow = row + xOffset[i]
            let newCol = col + yOffset[i]
            if newRow >= 0 && newRow < board.count && newCol >= 0 && newCol < board[0].count && "#" != board[newRow][newCol] && nil != currentNode && currentNode!.children.keys.contains(board[newRow][newCol]) {
                dfsFindWords(currentNode!, &board, newRow, newCol, &res)
            }
        }
        board[row][col] = letter
    }
    
    /**
     129. 求根到叶子节点数字之和
     
     给定一个二叉树，它的每个结点都存放一个 0-9 的数字，每条从根到叶子节点的路径都代表一个数字。

     例如，从根到叶子节点路径 1->2->3 代表数字 123。

     计算从根到叶子节点生成的所有数字之和。

     说明: 叶子节点是指没有子节点的节点。

     示例 1:

     输入: [1,2,3]
         1
        / \
       2   3
     输出: 25
     解释:
     从根到叶子节点路径 1->2 代表数字 12.
     从根到叶子节点路径 1->3 代表数字 13.
     因此，数字总和 = 12 + 13 = 25.
     示例 2:

     输入: [4,9,0,5,1]
         4
        / \
       9   0
      / \
     5   1
     输出: 1026
     解释:
     从根到叶子节点路径 4->9->5 代表数字 495.
     从根到叶子节点路径 4->9->1 代表数字 491.
     从根到叶子节点路径 4->0 代表数字 40.
     因此，数字总和 = 495 + 491 + 40 = 1026.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sum-root-to-leaf-numbers
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func sumNumbers(_ root: TreeNode?) -> Int {
        var res = [String]()
        dfsNumbers(root, "", &res)
        var sum = 0
        for str in res {
            sum += Int(str)!
        }
        return sum
    }
    
    private func dfsNumbers(_ root: TreeNode?, _ string: String, _ res: inout [String]) {
        if nil == root {
            return
        }
        
        if nil == root!.left && nil == root!.right {
            res.append(string + String(root!.val))
            return
        }
        
        dfsNumbers(root!.left, string + String(root!.val), &res)
        dfsNumbers(root!.right, string + String(root!.val), &res)
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
//        var str = String(n, radix: 2)
//        while str.count < 32 {
//            str = "0" + str
//        }
//        var res: Int = 0
//        let arr = Array(str)
//        for i in (0 ..< 32).reversed() {
//            if "1" == arr[i] {
//                res += Int(Float(powf(2, Float(i)))) * Int(String(arr[i]))!
//            }
//        }
//        return res
        
        var res = 0
        var tempN = n
        var index = 31
        while tempN > 0 {
            res += (tempN & 1) << index
            tempN = tempN >> 1
            index -= 1
        }
        return res
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
        
        var row = [Set<Character>](repeating: Set<Character>(), count: 9)
        var col = [Set<Character>](repeating: Set<Character>(), count: 9)
        var box = [Set<Character>](repeating: Set<Character>(), count: 9)
        for i in 0 ..< 9 {
            for j in 0 ..< 9 {
                let char = board[i][j]
                if "." == char {
                    continue
                }
                
                var rowSet = row[i]
                if rowSet.contains(char) {
                    return false
                }
                rowSet.insert(char)
                row[i] = rowSet
                
                var colSet = col[j]
                if colSet.contains(char) {
                    return false
                }
                colSet.insert(char)
                col[j] = colSet
                
                var boxSet = box[(i/3) * 3 + j/3]
                if boxSet.contains(char) {
                    return false
                }
                boxSet.insert(char)
                box[(i/3) * 3 + j/3] = boxSet
            }
        }
        
        return true
    }
    
    /**
     37. 解数独
     
     编写一个程序，通过填充空格来解决数独问题。

     一个数独的解法需遵循如下规则：

     数字 1-9 在每一行只能出现一次。
     数字 1-9 在每一列只能出现一次。
     数字 1-9 在每一个以粗实线分隔的 3x3 宫内只能出现一次。
     空白格用 '.' 表示。



     一个数独。



     答案被标成红色。

     提示：

     给定的数独序列只包含数字 1-9 和字符 '.' 。
     你可以假设给定的数独只有唯一解。
     给定数独永远是 9x9 形式的。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sudoku-solver
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func solveSudoku(_ board: inout [[Character]]) {
        guard 9 == board.count && 9 == board[0].count else {
            return
        }
        
        _ = dfsSloveSudoku(&board)
    }
    
    private func dfsSloveSudoku(_ board: inout [[Character]]) -> Bool {
        let numsArr = Array("123456789")
        for i in 0 ..< 9 {
            for j in 0 ..< 9 {
                if "." != board[i][j] {
                    continue
                }
                
                for num in numsArr {
                    if isValidSudoku(board, i, j, num) {
                        board[i][j] = num
                        if dfsSloveSudoku(&board) {
                            return true
                        }
                        board[i][j] = "."
                    }
                }
                
                return false
            }
        }
        
        return true
    }
    
    private func isValidSudoku(_ board: [[Character]], _ row: Int, _ col: Int, _ char: Character) -> Bool {
        for i in 0 ..< 9 {
            if "." != board[row][i] && char == board[row][i] {
                return false
            }
            
            if "." != board[i][col] && char == board[i][col] {
                return false
            }
            
            if "." != board[(row/3)*3 + i/3][3 * (col/3) + i%3] && char == board[(row/3)*3 + i/3][3 * (col/3) + i%3] {
                return false
            }
        }
        
        return true
    }
    
    /**
     463. 岛屿的周长
     
     给定一个包含 0 和 1 的二维网格地图，其中 1 表示陆地 0 表示水域。

     网格中的格子水平和垂直方向相连（对角线方向不相连）。整个网格被水完全包围，但其中恰好有一个岛屿（或者说，一个或多个表示陆地的格子相连组成的岛屿）。

     岛屿中没有“湖”（“湖” 指水域在岛屿内部且不和岛屿周围的水相连）。格子是边长为 1 的正方形。网格为长方形，且宽度和高度均不超过 100 。计算这个岛屿的周长。

      

     示例 :

     输入:
     [[0,1,0,0],
      [1,1,1,0],
      [0,1,0,0],
      [1,1,0,0]]

     输出: 16

     解释: 它的周长是下面图片中的 16 个黄色的边：




     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/island-perimeter
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func islandPerimeter(_ grid: [[Int]]) -> Int {
        let m = grid.count
        guard m > 0 else {
            return 0
        }
        
        let n = grid[0].count
        guard n > 0 else {
            return 0
        }
        
        var res = 0
        let xOffset = [-1, 0, 1, 0]
        let yOffset = [0, -1, 0, 1]
        for i in 0 ..< m {
            for j in 0 ..< n {
                if 1 == grid[i][j] {
                    for k in 0 ..< 4 {
                        let pos: (x: Int, y: Int) = (i + xOffset[k], j + yOffset[k])
                        switch k {
                        case 0:
                            if pos.x < 0 || 0 == grid[pos.x][pos.y] {
                                res += 1
                            }
                            
                        case 1:
                            if pos.y < 0 || 0 == grid[pos.x][pos.y] {
                                res += 1
                            }
                            
                        case 2:
                            if pos.x >= m || 0 == grid[pos.x][pos.y] {
                                res += 1
                            }
                            
                        default:
                            if pos.y >= n || 0 == grid[pos.x][pos.y] {
                                res += 1
                            }
                        }
                    }
                }
            }
        }
        return res
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
        
        let next = head?.next
        let nextnext = next?.next
        next?.next = head
        head?.next = swapPairs(nextnext)
        return next
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
        var res = 0
        for _ in 3 ... n {
            res = f1 + f2
            f1 = f2
            f2 = res
        }
        return res
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
    func generateParenthesis(_ n: Int) -> [String] {
        var res = [String]()
        generateParenthesisHelper(n, 0, 0, "", &res)
        return res
    }
    
    private func generateParenthesisHelper(_ n: Int, _ left: Int, _ right: Int, _ string: String, _ res: inout [String]) {
        if left == n && right == n {
            res.append(string)
            return
        }
        
        if left < n {
            generateParenthesisHelper(n, left + 1, right, string + "(", &res)
        }
        
        if right < left {
            generateParenthesisHelper(n, left, right + 1, string + ")", &res)
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
    func solveNQueens(_ n: Int) -> [[String]] {
        var res = [[String]]()
        guard n > 0 else {
            return res
        }
        
        var board = [[Character]]()
        for _ in 0 ..< n {
            board.append([Character](repeating: ".", count: n))
        }
        
        solveNQueuesBacktrack(n, &board, 0, &res)
        
        return res
    }
    
    private func solveNQueuesBacktrack(_ n: Int, _ board: inout [[Character]], _ row: Int, _ res: inout [[String]]) {
        if n == row {
            res.append(convertBoard(n, board))
            return
        }
        
        for i in 0 ..< n {
            if isValidNQueue(n, board, row, i) {
                board[row][i] = "Q"
                solveNQueuesBacktrack(n, &board, row + 1, &res)
                board[row][i] = "."
            }
        }
    }
    
    private func convertBoard(_ n: Int, _ board: [[Character]]) -> [String] {
        var res = [String]()
        for i in 0 ..< n {
            var string = ""
            for j in 0 ..< n {
                string += String(board[i][j])
            }
            res.append(string)
        }
        return res
    }
    
    private func isValidNQueue(_ n: Int, _ board: [[Character]], _ row: Int, _ col: Int) -> Bool {
        for i in 0 ..< n {
            // 行
            if "Q" == board[row][i] {
                return false
            }
            
            // 列
            if "Q" == board[i][col] {
                return false
            }
            
            // 左上角
            if row - i >= 0 && col - i >= 0 && "Q" == board[row - i][col - i] {
                return false
            }
                        
            // 右上角
            if row - i >= 0 && col + i < n && "Q" == board[row - i][col + i] {
                return false
            }
        }
        
        return true
    }
    
    /**
     237. 删除链表中的节点
     
     请编写一个函数，使其可以删除某个链表中给定的（非末尾）节点。传入函数的唯一参数为 要被删除的节点 。

      

     现有一个链表 -- head = [4,5,1,9]，它可以表示为:



      

     示例 1：

     输入：head = [4,5,1,9], node = 5
     输出：[4,1,9]
     解释：给定你链表中值为 5 的第二个节点，那么在调用了你的函数之后，该链表应变为 4 -> 1 -> 9.
     示例 2：

     输入：head = [4,5,1,9], node = 1
     输出：[4,5,9]
     解释：给定你链表中值为 1 的第三个节点，那么在调用了你的函数之后，该链表应变为 4 -> 5 -> 9.
      

     提示：

     链表至少包含两个节点。
     链表中所有节点的值都是唯一的。
     给定的节点为非末尾节点并且一定是链表中的一个有效节点。
     不要从你的函数中返回任何结果。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/delete-node-in-a-linked-list
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func deleteNode(_ node: ListNode?) {
        if nil != node && nil != node!.next {
            let val = node!.next!.val
            node!.val = val
            node!.next = node!.next!.next
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
     140. 单词拆分 II
     
     给定一个非空字符串 s 和一个包含非空单词列表的字典 wordDict，在字符串中增加空格来构建一个句子，使得句子中所有的单词都在词典中。返回所有这些可能的句子。

     说明：

     分隔时可以重复使用字典中的单词。
     你可以假设字典中没有重复的单词。
     示例 1：

     输入:
     s = "catsanddog"
     wordDict = ["cat", "cats", "and", "sand", "dog"]
     输出:
     [
       "cats and dog",
       "cat sand dog"
     ]
     示例 2：

     输入:
     s = "pineapplepenapple"
     wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
     输出:
     [
       "pine apple pen apple",
       "pineapple pen apple",
       "pine applepen apple"
     ]
     解释: 注意你可以重复使用字典中的单词。
     示例 3：

     输入:
     s = "catsandog"
     wordDict = ["cats", "dog", "sand", "and", "cat"]
     输出:
     []


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/word-break-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func wordBreak(_ s: String, _ wordDict: [String]) -> [String] {
        var res = [String]()
        let chars = Array(s)
        let wordSet = Set(wordDict)
        guard chars.count > 0 && wordSet.count > 0 else {
            return res
        }
        
        var memo = [Int: [[String]]]()
        let wordBreaks = wordBreakBacktrack(Array(s), s.count, wordSet, 0, &memo)
        for wordBreak in wordBreaks {
            res.append(wordBreak.joined(separator: " "))
        }
        
        return res
    }
    
    private func wordBreakBacktrack(_ chars: [Character], _ len: Int, _ wordSet: Set<String>, _ start: Int, _ memo: inout [Int: [[String]]]) -> [[String]] {
        if !memo.keys.contains(start) {
            var wordBreaks = [[String]]()
            if len == start {
                wordBreaks.append([String]())
                memo[start] = wordBreaks
                return memo[start]!
            }
            
            for i in start + 1 ... len {
                var word = ""
                for j in start ..< i {
                    word += String(chars[j])
                }
                if wordSet.contains(word) {
                    let nextWordBreaks = wordBreakBacktrack(chars, len, wordSet, i, &memo)
                    for nextWordBreak in nextWordBreaks {
                        var wordBreak = nextWordBreak
                        wordBreak.insert(word, at: 0)
                        wordBreaks.append(wordBreak)
                    }
                }
            }
            memo[start] = wordBreaks
        }
        
        return memo[start]!
    }
    
    /**
     718. 最长重复子数组
     
     给两个整数数组 A 和 B ，返回两个数组中公共的、长度最长的子数组的长度。

      

     示例：

     输入：
     A: [1,2,3,2,1]
     B: [3,2,1,4,7]
     输出：3
     解释：
     长度最长的公共子数组是 [3, 2, 1] 。
      

     提示：

     1 <= len(A), len(B) <= 1000
     0 <= A[i], B[i] < 100

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/maximum-length-of-repeated-subarray
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findLength(_ A: [Int], _ B: [Int]) -> Int {
        let m = A.count
        let n = B.count
        guard m > 0 && n > 0 else {
            return 0
        }
        
        return m < n ? findMaxLength(A, m, B, n) : findMaxLength(B, n, A, m)
    }
    
    private func findMaxLength(_ A: [Int], _ m: Int, _ B: [Int], _ n: Int) -> Int {
        var res = 0
        for i in 1 ... m {
            res = max(res, maxLength(A, 0, B, n - i, i))
        }
        for i in (0 ... n - m).reversed() {
            res = max(res, maxLength(A, 0, B, i, m))
        }
        for i in 1 ..< m {
            res = max(res, maxLength(A, i, B, 0, m - i))
        }
        return res
    }
    
    private func maxLength(_ A: [Int], _ i: Int, _ B: [Int], _ j: Int, _ len: Int) -> Int {
        var count = 0, res = 0
        for k in 0 ..< len {
            if A[k + i] == B[k + j] {
                count += 1
            } else if count > 0 {
                res = max(res, count)
                count = 0
            }
        }
        return count > 0 ? max(res, count) : res
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
        // 双向 BFS
        let wordSet = Set(wordList)
        guard wordSet.contains(endWord) else {
            return 0
        }
        var visited = Set<String>()
        
        var beginSet = Set<String>(), endSet = Set<String>()
        beginSet.insert(beginWord)
        endSet.insert(endWord)
        
        var len = 1
        let chars: [Character] = Array("abcdefghijklmnopqrstuvwxyz")
        while !beginSet.isEmpty && !endSet.isEmpty {
            if beginSet.count > endSet.count {
                let tempSet = endSet
                endSet = beginSet
                beginSet = tempSet
            }
            
            var tempSet = Set<String>()
            for word in beginSet {
                var wordArr = Array(word)
                for i in 0 ..< wordArr.count {
                    let currChar = wordArr[i]
                    for char in chars {
                        if currChar == char {
                            continue
                        }
                        
                        wordArr[i] = char
                        var nextWord = ""
                        for j in 0 ..< wordArr.count {
                            nextWord += String(wordArr[j])
                        }
                        if wordSet.contains(nextWord) {
                            if endSet.contains(nextWord) {
                                return len + 1
                            }
                            
                            if !visited.contains(nextWord) {
                                tempSet.insert(nextWord)
                                visited.insert(nextWord)
                            }
                        }
                    }
                    wordArr[i] = currChar
                }
            }
            len += 1
            beginSet = tempSet
        }
        return 0
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

      

     示例 1：

     输入：[[0,1],[1,0]]

     输出：2

     示例 2：

     输入：[[0,0,0],[1,1,0],[1,1,0]]

     输出：4

      

     提示：

     1 <= grid.length == grid[0].length <= 100
     grid[i][j] 为 0 或 1

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/shortest-path-in-binary-matrix
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func shortestPathBinaryMatrix(_ grid: [[Int]]) -> Int {
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
        
        // left - lefttop - top - righttop - right - rightdown - down - leftdown
        let xOffset = [-1, -1, 0, 1, 1, 1, 0, -1]
        let yOffset = [0, -1, -1, -1, 0, 1, 1, 1]
        var res = 1
        var queue = [(x: Int, y: Int)]()
        queue.append((0, 0))
        var visited = [[Bool]]()
        for _ in 0 ..< m {
            visited.append([Bool](repeating: false, count: n))
        }
        visited[0][0] = true
        while !queue.isEmpty {
            let size = queue.count
            for _ in 0 ..< size {
                let loc = queue.removeFirst()
                if m - 1 == loc.x && n - 1 == loc.y {
                    return res
                }
                
                for i in 0 ..< 8 {
                    let newX = loc.x + xOffset[i]
                    let newY = loc.y + yOffset[i]
                    if newX >= 0 && newX < m && newY >= 0 && newY < n && 0 == grid[newX][newY] && !visited[newX][newY] {
                        queue.append((newX, newY))
                        visited[newX][newY] = true
                    }
                }
            }
            res += 1
        }
        return -1
    }
    
    /**
     773. 滑动谜题
     
     在一个 2 x 3 的板上（board）有 5 块砖瓦，用数字 1~5 来表示, 以及一块空缺用 0 来表示.

     一次移动定义为选择 0 与一个相邻的数字（上下左右）进行交换.

     最终当板 board 的结果是 [[1,2,3],[4,5,0]] 谜板被解开。

     给出一个谜板的初始状态，返回最少可以通过多少次移动解开谜板，如果不能解开谜板，则返回 -1 。

     示例：

     输入：board = [[1,2,3],[4,0,5]]
     输出：1
     解释：交换 0 和 5 ，1 步完成
     输入：board = [[1,2,3],[5,4,0]]
     输出：-1
     解释：没有办法完成谜板
     输入：board = [[4,1,2],[5,0,3]]
     输出：5
     解释：
     最少完成谜板的最少移动次数是 5 ，
     一种移动路径:
     尚未移动: [[4,1,2],[5,0,3]]
     移动 1 次: [[4,1,2],[0,5,3]]
     移动 2 次: [[0,1,2],[4,5,3]]
     移动 3 次: [[1,0,2],[4,5,3]]
     移动 4 次: [[1,2,0],[4,5,3]]
     移动 5 次: [[1,2,3],[4,5,0]]
     输入：board = [[3,2,4],[1,5,0]]
     输出：14
     提示：

     board 是一个如上所述的 2 x 3 的数组.
     board[i][j] 是一个 [0, 1, 2, 3, 4, 5] 的排列.

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/sliding-puzzle
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func slidingPuzzle(_ board: [[Int]]) -> Int {
        let m = board.count
        guard m > 0 else {
            return -1
        }
        
        let n = board[0].count
        guard n > 0 else {
            return -1
        }
        
        var arr = [String]()
        for i in 0 ..< m {
            for j in 0 ..< n {
                arr.append(String(board[i][j]))
            }
        }
        
        let moves = [0: [1, 3], 1: [0, 2, 4], 2: [1, 5], 3: [0, 4], 4: [1, 3, 5], 5: [2, 4]]
        
        var count = 0
        var used = Set<String>()
        var queue = [String]()
        queue.append(arr.joined())
        while !queue.isEmpty {
            let size = queue.count
            for _ in 0 ..< size {
                let string = queue.removeFirst()
                used.insert(string)
                if string == "123450" {
                    return count
                }
                
                var index = 0
                let stringArr = Array(string)
                for i in 0 ..< stringArr.count {
                    arr[i] = String(stringArr[i])
                    if "0" == stringArr[i] {
                        index = i
                    }
                }
                for move in moves[index]! {
                    var newArr = arr
                    let str = newArr[index]
                    newArr[index] = newArr[move]
                    newArr[move] = str
                    let newString = newArr.joined()
                    if !used.contains(newString) {
                        queue.append(newString)
                    }
                }
            }
            count += 1
        }
        return -1
    }
    
    /**
     433. 最小基因变化
     
     一条基因序列由一个带有8个字符的字符串表示，其中每个字符都属于 "A", "C", "G", "T"中的任意一个。

     假设我们要调查一个基因序列的变化。一次基因变化意味着这个基因序列中的一个字符发生了变化。

     例如，基因序列由"AACCGGTT" 变化至 "AACCGGTA" 即发生了一次基因变化。

     与此同时，每一次基因变化的结果，都需要是一个合法的基因串，即该结果属于一个基因库。

     现在给定3个参数 — start, end, bank，分别代表起始基因序列，目标基因序列及基因库，请找出能够使起始基因序列变化为目标基因序列所需的最少变化次数。如果无法实现目标变化，请返回 -1。

     注意:

     起始基因序列默认是合法的，但是它并不一定会出现在基因库中。
     所有的目标基因序列必须是合法的。
     假定起始基因序列与目标基因序列是不一样的。
     示例 1:

     start: "AACCGGTT"
     end:   "AACCGGTA"
     bank: ["AACCGGTA"]

     返回值: 1
     示例 2:

     start: "AACCGGTT"
     end:   "AAACGGTA"
     bank: ["AACCGGTA", "AACCGCTA", "AAACGGTA"]

     返回值: 2
     示例 3:

     start: "AAAAACCC"
     end:   "AACCCCCC"
     bank: ["AAAACCCC", "AAACCCCC", "AACCCCCC"]

     返回值: 3

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/minimum-genetic-mutation
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func minMutation(_ start: String, _ end: String, _ bank: [String]) -> Int {
        guard start != end else {
            return 0
        }
        
        let bankSet = Set(bank)
        guard bankSet.contains(end) else {
            return -1
        }
        
        var startSet = Set<String>(), endSet = Set<String>()
        startSet.insert(start)
        endSet.insert(end)
        
        var visited = Set<String>()
        visited.insert(start)
        visited.insert(end)
        
        var res = 1
        let chars: [Character] = Array("ACGT")
        while !startSet.isEmpty && !endSet.isEmpty {
            if startSet.count > endSet.count {
                let temp = startSet
                startSet = endSet
                endSet = temp
            }
            
            var tempSet = Set<String>()
            for word in startSet {
                var wordArr = Array(word)
                for i in 0 ..< wordArr.count {
                    let char = wordArr[i]
                    for c in chars {
                        if c == char {
                            continue
                        }
                        
                        wordArr[i] = c
                        var nextWord = ""
                        for j in 0 ..< wordArr.count {
                            nextWord += String(wordArr[j])
                        }
                        if endSet.contains(nextWord) {
                            return res
                        }
                        if bankSet.contains(nextWord) && !visited.contains(nextWord) {
                            tempSet.insert(nextWord)
                            visited.insert(nextWord)
                        }
                    }
                    wordArr[i] = char
                }
            }
            
            res += 1
            startSet = tempSet
        }
        return -1
    }
}

class UnionFind {
    private var count: Int = 0
    private var parent: [Int]
    
    init(_ n: Int) {
        count = n
        parent = [Int](repeating: 0, count: n)
        for i in 0 ..< n {
            parent[i] = i
        }
    }
    
    func find(_ p: Int) -> Int {
        var x = p
        while x != parent[x] {
            parent[x] = parent[parent[x]]
            x = parent[x]
        }
        return x
    }
    
    func union(_ p: Int, _ q: Int) {
        let rootP = find(p)
        let rootQ = find(q)
        if rootP == rootQ {
            return
        }
        parent[rootP] = rootQ
        count -= 1
    }
    
    func getCount() -> Int {
        return count
    }
    
    func connected(_ p: Int, _ q: Int) -> Bool {
        let rootP = find(p)
        let rootQ = find(q)
        return rootP == rootQ
    }
}

/**
 381. O(1) 时间插入、删除和获取随机元素 - 允许重复
 
 设计一个支持在平均 时间复杂度 O(1) 下， 执行以下操作的数据结构。

 注意: 允许出现重复元素。

 insert(val)：向集合中插入元素 val。
 remove(val)：当 val 存在时，从集合中移除一个 val。
 getRandom：从现有集合中随机获取一个元素。每个元素被返回的概率应该与其在集合中的数量呈线性相关。
 示例:

 // 初始化一个空的集合。
 RandomizedCollection collection = new RandomizedCollection();

 // 向集合中插入 1 。返回 true 表示集合不包含 1 。
 collection.insert(1);

 // 向集合中插入另一个 1 。返回 false 表示集合包含 1 。集合现在包含 [1,1] 。
 collection.insert(1);

 // 向集合中插入 2 ，返回 true 。集合现在包含 [1,1,2] 。
 collection.insert(2);

 // getRandom 应当有 2/3 的概率返回 1 ，1/3 的概率返回 2 。
 collection.getRandom();

 // 从集合中删除 1 ，返回 true 。集合现在包含 [1,2] 。
 collection.remove(1);

 // getRandom 应有相同概率返回 1 和 2 。
 collection.getRandom();

 来源：力扣（LeetCode）
 链接：https://leetcode-cn.com/problems/insert-delete-getrandom-o1-duplicates-allowed
 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 */
class RandomizedCollection {
    private var nums: [Int] = [Int]()
    private var idx: [Int: Set<Int>] = [:]
    
    /** Initialize your data structure here. */
    init() {
        
    }
    
    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    func insert(_ val: Int) -> Bool {
        nums.append(val)
        if var set = idx[val] {
            let hasElement = set.count > 0
            set.insert(nums.count - 1)
            idx[val] = set
            return !hasElement
        } else {
            var set = Set<Int>()
            set.insert(nums.count - 1)
            idx[val] = set
            return true
        }
    }
    
    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    func remove(_ val: Int) -> Bool {
        if !idx.keys.contains(val) {
            return false
        }
        
        var iter = idx[val]!.makeIterator()
        let i = iter.next()!
        let lastNum = nums[nums.count - 1]
        nums[i] = lastNum
        idx[val]!.remove(i)
        idx[lastNum]!.remove(nums.count - 1)
        if i < nums.count - 1 {
            idx[lastNum]!.insert(i)
        }
        if 0 == idx[val]!.count {
            idx.removeValue(forKey: val)
        }
        nums.removeLast()
        return true
    }
    
    /** Get a random element from the collection. */
    func getRandom() -> Int {
        if nums.count > 0 {
            return nums[Int(random())%nums.count]
        }
        
        return 0
    }
    
    static var state: UInt32 = 1
    private func random() -> UInt32 {
        var x: UInt32 = RandomizedCollection.state
        x = x^(x << 13)
        x = x^(x >> 17)
        x = x^(x << 5)
        RandomizedCollection.state = x
        return x
    }
}
