# 学习笔记

今天的每日一题，`1024. 视频拼接` ，竟然在没有看题解的情况下，自己解出来了，想来这段时间的练习，还是有所收获，提交通过的那一刻，自己真的是兴奋坏了。

这次就以自己对该题的理解作为一篇学习笔记，分析下自己对题目思考的过程。

题目如下：

```
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
```

由于做过很多次单次接龙的题目，看到该题是求解最小片段数，第一时间就想到了 BFS，找到 `clips` 中 `clips[i][1]` 最大的那个 `clip` 元素 `maxClip`，首先判断是否 `maxClip[0] <= 0 && maxClip[1] >= T`，是的话，则直接返回 1， 然后以 `maxClip` 作为树的根节点，通过 BFS 找到一个 `clip[1] >= maxClip[0]` 的元素作为子树，若该元素的 `clip[0] <= 0 `，则已经能得出正确结果，即遍历到了子节点，遍历结束，找到最短路径，返回结果即可，代码大体框架跟单词接龙的框架类似。

执行完例题提供的四个测试 case，发现都通过了，信心满满，以为自己已经完成，结果一提交，发现并未通过，再根据测试出错的 case 一分析，发现除了 BFS，还需要回溯，对于每一个 `clip[i][1] >= T` 的元素，都需要将其作为根节点进行 BFS，然后在所有最短路径中再取最小值。于是，修改代码，重新测试 case，重新提交，通过。

代码还有很大的改进空间，但是最起码自己是真的进步了，放以前，在参加算法训练营之前，这种题目，我是绝对做不出来的，多半连一点思路都没有。所以后续一定要坚持练习，坚持使用“五毒神掌”的方法进行训练，过遍数真的真的非常重要，过的次数多了，真的就慢慢理解了。比如三数之和的题目中，之前，总是理解不了去重的那个逻辑，但是，自从上次做了四数之和之和，今天再做三数之和就完全理解了。所谓“五毒神掌”，就是指“书读百遍其义自见”。

附 `1024. 视频拼接` `Swift` 代码：

```swift
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
```