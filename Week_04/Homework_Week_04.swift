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

class Homework_Week_04 {
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
        // DFS(回溯)， 将 start 与 bank 进行对比，若 start 通过一步改变就能变成 bank 中的某一个元素，则 minChange + 1，若最终 start 能变成跟 end 一样，则返回 minChange，否则返回 -1
        var res = Int.max
        var bankArr = [[Character]]()
        var visited = [Bool]()
        for str in bank {
            bankArr.append(Array(str))
            visited.append(false)
        }
        let startArr = Array(start)
        let endArr = Array(end)
        minMutationBacktrack(startArr, endArr, bankArr, 0, &res, &visited)
        return Int.max == res ? -1 : res
    }
    
    private func minMutationBacktrack(_ start: [Character], _ end: [Character], _ bank: [[Character]], _ change: Int, _ res: inout Int, _ visited: inout [Bool]) {
        if start == end {
            res = min(change, res)
            return
        }
        
        for i in 0 ..< bank.count {
            if visited[i] {
                continue
            }
            
            let bankElement = bank[i]
            var diff = 0
            for j in 0 ..< start.count {
                if start[j] != bankElement[j] {
                    diff += 1
                }
            }
            
            if 1 == diff {
                visited[i] = true
                minMutationBacktrack(bankElement, end, bank, change + 1, &res, &visited)
                visited[i] = false
            }
        }
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
     515. 在每个树行中找最大值
     
     您需要在二叉树的每一行中找到最大的值。

     示例：

     输入:

               1
              / \
             3   2
            / \   \
           5   3   9

     输出: [1, 3, 9]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func largestValues(_ root: TreeNode?) -> [Int] {
        var res = [Int]()
        var queue = [TreeNode]()
        if let node = root {
            queue.append(node)
        }
        while !queue.isEmpty {
            let size = queue.count
            var maxNum = Int.min
            for _ in 0 ..< size {
                let node = queue.removeFirst()
                maxNum = max(maxNum, node.val)
                if let left = node.left {
                    queue.append(left)
                }
                if let right = node.right {
                    queue.append(right)
                }
            }
            res.append(maxNum)
        }
        return res
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
        if 0 == wordList.count || !wordList.contains(endWord) {
            return 0
        }
        
        var wordSet = Set(wordList)
        wordSet.remove(beginWord)
        
        var queue = [String]()
        queue.append(beginWord)
        
        var visited = Set<String>()
        visited.insert(beginWord)
        
        var step = 1
        while !queue.isEmpty {
            let size = queue.count
            for _ in 0 ..< size {
                let currentWord = queue.removeFirst()
                if changeWordEveryOneLetter(currentWord, endWord, &queue, &visited, wordSet) {
                    return step + 1
                }
            }
            step += 1
        }
        
        return 0
    }
    
    private let characters: [Character] = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    private func changeWordEveryOneLetter(_ currentWord: String, _ endWord: String, _ queue: inout [String], _ visited: inout Set<String>, _ wordSet: Set<String>) -> Bool {
        var currentWordArr = Array(currentWord)
        for i in 0 ..< endWord.count {
            let char = currentWordArr[i]
            for c in characters {
                if char == c {
                    continue
                }
                
                currentWordArr[i] = c
                let nextWord = currentWordArr.compactMap { "\($0)" }.joined()
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
            currentWordArr[i] = char
        }
        return false
    }
    
    /**
     126. 单词接龙 II
     
     给定两个单词（beginWord 和 endWord）和一个字典 wordList，找出所有从 beginWord 到 endWord 的最短转换序列。转换需遵循如下规则：

     每次转换只能改变一个字母。
     转换后得到的单词必须是字典中的单词。
     说明:

     如果不存在这样的转换序列，返回一个空列表。
     所有单词具有相同的长度。
     所有单词只由小写字母组成。
     字典中不存在重复的单词。
     你可以假设 beginWord 和 endWord 是非空的，且二者不相同。
     示例 1:

     输入:
     beginWord = "hit",
     endWord = "cog",
     wordList = ["hot","dot","dog","lot","log","cog"]

     输出:
     [
       ["hit","hot","dot","dog","cog"],
       ["hit","hot","lot","log","cog"]
     ]
     示例 2:

     输入:
     beginWord = "hit"
     endWord = "cog"
     wordList = ["hot","dot","dog","lot","log"]

     输出: []

     解释: endWord "cog" 不在字典中，所以不存在符合要求的转换序列。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/word-ladder-ii
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findLadders(_ beginWord: String, _ endWord: String, _ wordList: [String]) -> [[String]] {
        var res = [[String]]()
        if 0 == beginWord.count || beginWord.count != endWord.count || !wordList.contains(endWord) {
            return res
        }
        
        findLaddersBFS(beginWord, endWord, wordList, &res)
        return res
    }
    
    private func findLaddersBFS(_ beginWord: String, _ endWord: String, _ wordList: [String], _ res: inout [[String]]) {
        var queue = [[String]]()
        var path = [String]()
        path.append(beginWord)
        queue.append(path)
        var isFound = false
        let wordSet = Set(wordList)
        var visited = Set<String>()
        visited.insert(beginWord)
        while !queue.isEmpty {
            let size = queue.count
            var subVisited = Set<String>()
            for _ in 0 ..< size {
                var currentPath = queue.removeFirst()
                let lastWordInCurrentPath = currentPath.last!
                let neighbors = getNeighbors(lastWordInCurrentPath, wordSet)
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        if neighbor == endWord {
                            isFound = true
                            currentPath.append(neighbor)
                            res.append(currentPath)
                            currentPath.removeLast()
                        }
                        
                        currentPath.append(neighbor)
                        queue.append(currentPath)
                        currentPath.removeLast()
                        subVisited.insert(neighbor)
                    }
                }
            }
            
            visited = visited.union(subVisited)
            
            if isFound {
                break
            }
        }
    }
    
    private var getNeighborsInterval = 0.0
    private func getNeighbors(_ word: String, _ wordSet: Set<String>) -> [String] {
        var res = [String]()
        var wordArr = Array(word)
        for i in 0 ..< wordArr.count {
            let originChar = wordArr[i]
            for char in characters {
                if char == originChar {
                    continue
                }
                
                wordArr[i] = char
                
                
                var neighborWord = ""
                for c in wordArr {
                    neighborWord += "\(c)"
                }
                
                if wordSet.contains(neighborWord) {
                    res.append(neighborWord)
                }
                
            }
            
            wordArr[i] = originChar
        }
        
        return res
    }
    
    /**
     200. 岛屿数量
     
     给你一个由 '1'（陆地）和 '0'（水）组成的的二维网格，请你计算网格中岛屿的数量。

     岛屿总是被水包围，并且每座岛屿只能由水平方向或竖直方向上相邻的陆地连接形成。

     此外，你可以假设该网格的四条边均被水包围。

      

     示例 1:

     输入:
     [
     ['1','1','1','1','0'],
     ['1','1','0','1','0'],
     ['1','1','0','0','0'],
     ['0','0','0','0','0']
     ]
     输出: 1
     示例 2:

     输入:
     [
     ['1','1','0','0','0'],
     ['1','1','0','0','0'],
     ['0','0','1','0','0'],
     ['0','0','0','1','1']
     ]
     输出: 3
     解释: 每座岛屿只能由水平和/或竖直方向上相邻的陆地连接而成。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/number-of-islands
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func numIslands(_ grid: [[Character]]) -> Int {
        let row = grid.count
        if 0 == row {
            return 0
        }
        let col = grid[0].count
        var count = 0
        var tmpGrid = grid
        for i in 0 ..< row {
            for j in 0 ..< col {
                if tmpGrid[i][j] == "1" {
                    dfsMarkIslands(i, j, row, col, &tmpGrid)
                    count += 1
                }
            }
        }
        return count
    }
    
    private func dfsMarkIslands(_ i: Int, _ j: Int, _ row: Int, _ col: Int, _ grid: inout [[Character]]) {
        if i < 0 || i >= row || j < 0 || j >= col || grid[i][j] != "1" {
            return
        }
                
        grid[i][j] = "0"
        dfsMarkIslands(i - 1, j, row, col, &grid)
        dfsMarkIslands(i + 1, j, row, col, &grid)
        dfsMarkIslands(i, j - 1, row, col, &grid)
        dfsMarkIslands(i, j + 1, row, col, &grid)
    }
    
    /**
     529. 扫雷游戏
     
     让我们一起来玩扫雷游戏！

     给定一个代表游戏板的二维字符矩阵。 'M' 代表一个未挖出的地雷，'E' 代表一个未挖出的空方块，'B' 代表没有相邻（上，下，左，右，和所有4个对角线）地雷的已挖出的空白方块，数字（'1' 到 '8'）表示有多少地雷与这块已挖出的方块相邻，'X' 则表示一个已挖出的地雷。

     现在给出在所有未挖出的方块中（'M'或者'E'）的下一个点击位置（行和列索引），根据以下规则，返回相应位置被点击后对应的面板：

     如果一个地雷（'M'）被挖出，游戏就结束了- 把它改为 'X'。
     如果一个没有相邻地雷的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的未挖出方块都应该被递归地揭露。
     如果一个至少与一个地雷相邻的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量。
     如果在此次点击中，若无更多方块可被揭露，则返回面板。
      

     示例 1：

     输入:

     [['E', 'E', 'E', 'E', 'E'],
      ['E', 'E', 'M', 'E', 'E'],
      ['E', 'E', 'E', 'E', 'E'],
      ['E', 'E', 'E', 'E', 'E']]

     Click : [3,0]

     输出:

     [['B', '1', 'E', '1', 'B'],
      ['B', '1', 'M', '1', 'B'],
      ['B', '1', '1', '1', 'B'],
      ['B', 'B', 'B', 'B', 'B']]

     解释:

     示例 2：

     输入:

     [['B', '1', 'E', '1', 'B'],
      ['B', '1', 'M', '1', 'B'],
      ['B', '1', '1', '1', 'B'],
      ['B', 'B', 'B', 'B', 'B']]

     Click : [1,2]

     输出:

     [['B', '1', 'E', '1', 'B'],
      ['B', '1', 'X', '1', 'B'],
      ['B', '1', '1', '1', 'B'],
      ['B', 'B', 'B', 'B', 'B']]

     解释:

      

     注意：

     输入矩阵的宽和高的范围为 [1,50]。
     点击的位置只能是未被挖出的方块 ('M' 或者 'E')，这也意味着面板至少包含一个可点击的方块。
     输入面板不会是游戏结束的状态（即有地雷已被挖出）。
     简单起见，未提及的规则在这个问题中可被忽略。例如，当游戏结束时你不需要挖出所有地雷，考虑所有你可能赢得游戏或标记方块的情况。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/minesweeper
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func updateBoard(_ board: [[Character]], _ click: [Int]) -> [[Character]] {
        guard 2 == click.count else {
            return board
        }
        
        let row = board.count
        guard row > 0 else {
            return board
        }
        
        let col = board[0].count
        let clickRow = click[0]
        let clickCol = click[1]
        guard clickRow >= 0 && clickRow < row && clickCol >= 0 && clickCol < col else {
            return board
        }
        
        var res = board
        if board[clickRow][clickCol] == "M" {
            res[clickRow][clickCol] = "X"
        } else {
            dfsUpdateBoard(&res, clickRow, clickCol, row, col)
        }
        return res
    }
    
    private let xBoard = [-1, -1, -1, 0, 1, 1, 1, 0]
    private let yBoard = [-1, 0, 1, 1, 1, 0, -1, -1]
    private func dfsUpdateBoard(_ board: inout [[Character]], _ row: Int, _ col: Int, _ boardRow: Int, _ boardCol: Int) {
        // 查找 board[row][col] 元素的上下左右及四个对角线共八个位置中地雷的数量
        var mCount: Int = 0
        for i in 0 ..< 8 {
            // 从左上角开始，逆时针扫描
            let x = row + xBoard[i]
            let y = col + yBoard[i]
            guard x >= 0 && x < boardRow && y >= 0 && y < boardCol else {
                continue
            }
            
            if board[x][y] == "M" { // 找到地雷，数量加 1
                mCount += 1
            }
        }
        
        if mCount > 0 {
            // 规则3，如果一个至少与一个地雷相邻的空方块（'E'）被挖出，修改它为数字（'1'到'8'），表示相邻地雷的数量
            board[row][col] = Array("\(mCount)")[0]
        } else {
            // 规则2，如果一个没有相邻地雷的空方块（'E'）被挖出，修改它为（'B'），并且所有和其相邻的未挖出方块都应该被递归地揭露
            board[row][col] = "B"
            for i in 0 ..< 8 {
                // 从左上角开始，逆时针扫描
                let x = row + xBoard[i]
                let y = col + yBoard[i]
                guard x >= 0 && x < boardRow && y >= 0 && y < boardCol && board[x][y] == "E" else {
                    continue
                }
                
                dfsUpdateBoard(&board, x, y, boardRow, boardCol)
            }
        }
    }
    
    /**
     860. 柠檬水找零
     
     在柠檬水摊上，每一杯柠檬水的售价为 5 美元。

     顾客排队购买你的产品，（按账单 bills 支付的顺序）一次购买一杯。

     每位顾客只买一杯柠檬水，然后向你付 5 美元、10 美元或 20 美元。你必须给每个顾客正确找零，也就是说净交易是每位顾客向你支付 5 美元。

     注意，一开始你手头没有任何零钱。

     如果你能给每位顾客正确找零，返回 true ，否则返回 false 。

     示例 1：

     输入：[5,5,5,10,20]
     输出：true
     解释：
     前 3 位顾客那里，我们按顺序收取 3 张 5 美元的钞票。
     第 4 位顾客那里，我们收取一张 10 美元的钞票，并返还 5 美元。
     第 5 位顾客那里，我们找还一张 10 美元的钞票和一张 5 美元的钞票。
     由于所有客户都得到了正确的找零，所以我们输出 true。
     示例 2：

     输入：[5,5,10]
     输出：true
     示例 3：

     输入：[10,10]
     输出：false
     示例 4：

     输入：[5,5,10,10,20]
     输出：false
     解释：
     前 2 位顾客那里，我们按顺序收取 2 张 5 美元的钞票。
     对于接下来的 2 位顾客，我们收取一张 10 美元的钞票，然后返还 5 美元。
     对于最后一位顾客，我们无法退回 15 美元，因为我们现在只有两张 10 美元的钞票。
     由于不是每位顾客都得到了正确的找零，所以答案是 false。
      

     提示：

     0 <= bills.length <= 10000
     bills[i] 不是 5 就是 10 或是 20


     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/lemonade-change
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func lemonadeChange(_ bills: [Int]) -> Bool {
        if 0 == bills.count {
            return true
        }
        
        if 10 == bills[0] {
            return false
        }
        
        var count5 = 1
        var count10 = 0
        for i in 1 ..< bills.count {
            if 5 == bills[i] {
                count5 += 1
            } else if 10 == bills[i] {
                if 0 == count5 {
                    return false
                }
                
                count5 -= 1
                count10 += 1
            } else {
                if 0 == count5 {
                    return false
                }
                
                if 0 == count10 {
                    if count5 < 3 {
                        return false
                    }
                    count5 -= 3
                } else {
                    count5 -= 1
                    count10 -= 1
                }
            }
        }
        
        return true
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
        // rightmost 为能够到达的最右边的索引
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
        let n = nums.count
        var maxPosition = 0 // 能跳到的最远位置
        var end = 0         // 下标更新到该最远位置
        var steps = 0       // 步数
        for i in 0 ..< n - 1 {
            maxPosition = max(maxPosition, i + nums[i])
            if i == end {
                end = maxPosition
                steps += 1
            }
        }
        return steps
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
     874. 模拟行走机器人
     
     机器人在一个无限大小的网格上行走，从点 (0, 0) 处开始出发，面向北方。该机器人可以接收以下三种类型的命令：

     -2：向左转 90 度
     -1：向右转 90 度
     1 <= x <= 9：向前移动 x 个单位长度
     在网格上有一些格子被视为障碍物。

     第 i 个障碍物位于网格点  (obstacles[i][0], obstacles[i][1])

     机器人无法走到障碍物上，它将会停留在障碍物的前一个网格方块上，但仍然可以继续该路线的其余部分。

     返回从原点到机器人所有经过的路径点（坐标为整数）的最大欧式距离的平方。

      

     示例 1：

     输入: commands = [4,-1,3], obstacles = []
     输出: 25
     解释: 机器人将会到达 (3, 4)
     示例 2：

     输入: commands = [4,-1,4,-2,4], obstacles = [[2,4]]
     输出: 65
     解释: 机器人在左转走到 (1, 8) 之前将被困在 (1, 4) 处
      

     提示：

     0 <= commands.length <= 10000
     0 <= obstacles.length <= 10000
     -30000 <= obstacle[i][0] <= 30000
     -30000 <= obstacle[i][1] <= 30000
     答案保证小于 2 ^ 31

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/walking-robot-simulation
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    */
    func robotSim(_ commands: [Int], _ obstacles: [[Int]]) -> Int {
        let robotX = [0, 1, 0, -1]
        let robotY = [1, 0, -1, 0]
        var finalX = 0
        var finalY = 0
        
        var obstacleSet = Set<Int>()
        for obstacle in obstacles {
            if 2 == obstacle.count {
                let ox = obstacle[0] + 30000
                let oy = obstacle[1] + 30000
                obstacleSet.insert((ox << 16) + oy)
            }
        }
        
        var res = 0
        var direction = 0 // 0 - 向上，1 - 向右，2 - 向下，3 - 向左
        for command in commands {
            if -1 == command {
                direction = (direction + 1)%4
            } else if -2 == command {
                direction = (direction + 3)%4
            } else {
                for _ in 0 ..< command {
                    let nx = finalX + robotX[direction]
                    let ny = finalY + robotY[direction]
                    let code = ((nx + 30000) << 16) + ny + 30000
                    if !obstacleSet.contains(code) {
                        finalX = nx
                        finalY = ny
                        res = max(res, finalX * finalX + finalY * finalY) % 1000000007
                    }
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
        var cur = 1.0
        while true {
            let pre = cur
            cur = (cur + Double(num)/cur) / 2
            if fabs(pre - cur) < 0.000001 {
                break
            }
        }
        return fabs(cur - Double(Int(cur))) < 0.000001
    }
    
    /**
     33. 搜索旋转排序数组
     
     假设按照升序排序的数组在预先未知的某个点上进行了旋转。

     ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

     搜索一个给定的目标值，如果数组中存在这个目标值，则返回它的索引，否则返回 -1 。

     你可以假设数组中不存在重复的元素。

     你的算法时间复杂度必须是 O(log n) 级别。

     示例 1:

     输入: nums = [4,5,6,7,0,1,2], target = 0
     输出: 4
     示例 2:

     输入: nums = [4,5,6,7,0,1,2], target = 3
     输出: -1

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/search-in-rotated-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func search(_ nums: [Int], _ target: Int) -> Int {
        var left = 0, right = nums.count - 1, mid = 0
        while left <= right {
            mid = left + (right - left) >> 1
            if target == nums[mid] {
                return mid
            }
            
            if nums[mid] >= nums[left] {    // mid 在左段
                if target >= nums[left] && target <= nums[mid] { // target 在 mid 的左边
                    right = mid - 1
                } else {
                    left = mid + 1
                }
            } else {    // mid 在右段
                if target > nums[mid] && target <= nums[right] {
                    left = mid + 1
                } else {
                    right = mid - 1
                }
            }
        }
        return -1
    }
    
    /**
     153. 寻找旋转排序数组中的最小值
     
     假设按照升序排序的数组在预先未知的某个点上进行了旋转。

     ( 例如，数组 [0,1,2,4,5,6,7] 可能变为 [4,5,6,7,0,1,2] )。

     请找出其中最小的元素。

     你可以假设数组中不存在重复元素。

     示例 1:

     输入: [3,4,5,1,2]
     输出: 1
     示例 2:

     输入: [4,5,6,7,0,1,2]
     输出: 0

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func findMin(_ nums: [Int]) -> Int {
        var left = 0, right = nums.count - 1, mid = 0
        if 1 == nums.count {
            return nums[0]
        }
        if nums[0] < nums[right] {
            return nums[0]
        }
        if 2 == nums.count {
            return min(nums[0], nums[1])
        }
        while left <= right {
            mid = left + (right - left)/2

            if nums[mid] > nums[mid + 1] {
                return nums[mid + 1]
            }
            
            if nums[mid] < nums[mid - 1] {
                return nums[mid]
            }
            
            if nums[mid] > nums[0] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return -1
    }
    
    /**
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
        
        // 从右侧顶点开始搜索
        var x = 0
        var y = n - 1
        while x < m && y >= 0 {
            if target == matrix[x][y] {
                return true
            } else if matrix[x][y] > target {   // 去掉第 y 列
                y -= 1
            } else {                            // 去掉第 x 行
                x += 1
            }
        }
        
        return false
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
        let m = text1.count
        let n = text2.count
        guard m > 0 && n > 0 else {
            return 0
        }
        
        // dp[i][j] 表示 text1[0 ... i] 与 text2[0 ... j] 的最长公共子序列
        var dp = [[Int]]()
        for _ in 0 ... m {
            dp.append([Int](repeating: 0, count: n + 1))
        }
        
        let chars1 = Array(text1)
        let chars2 = Array(text2)
        for i in 1 ... m {
            for j in 1 ... n {
                if chars1[i - 1] == chars2[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }
        
        return dp[m][n]
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
//    func minimumTotal(_ triangle: [[Int]]) -> Int {
//        let n = triangle.count
//        guard n > 0 else {
//            return Int.min
//        }
//
//        // dp[i][j] 表示从点 (i, j) 到底边的最小路径和
//        var dp = [[Int]]()
//        for _ in 0 ... n {
//            dp.append([Int](repeating: 0, count: n + 1))
//        }
//        for i in (0 ..< n).reversed() {
//            for j in 0 ..< i + 1 {
//                dp[i][j] = triangle[i][j] + min(dp[i + 1][j], dp[i + 1][j + 1])
//            }
//        }
//        return dp[0][0]
//    }
    
    func minimumTotal(_ triangle: [[Int]]) -> Int {
        let n = triangle.count
        guard n > 0 else {
            return Int.min
        }
        
        var dp = [Int](repeating: 0, count: n + 1)
        for i in (0 ..< n).reversed() {
            for j in 0 ... i {
                dp[j] = min(dp[j], dp[j + 1]) + triangle[i][j]
            }
        }
        return dp[0]
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
        
        let n = nums.count
        // dp[i] 表示以第 i 个数结尾的「连续子数组的最大和」
        var dp = [Int](repeating: 0, count: n)
        dp[0] = nums[0]
        for i in 1 ..< n {
            dp[i] = max(nums[i], dp[i - 1] + nums[i])
        }
        
        var res = dp[0]
        for i in 1 ..< n {
            res = max(res, dp[i])
        }
        
        return res
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
        // dp[i] 表示到第 i 号房间能偷窃到的最高金额
        var dp = [Int](repeating: 0, count: n)
        dp[0] = nums[0]
        dp[1] = max(nums[0], nums[1])
        for i in 2 ..< n {
            dp[i] = max(nums[i] + dp[i - 2], dp[i - 1])
        }
        
        return dp[n - 1]
    }
}
