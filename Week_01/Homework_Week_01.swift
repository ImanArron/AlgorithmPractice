//
//  SlideWindow.swift
//  DataStructDemo
//
//  Created by 刘练 on 2020/9/6.
//  Copyright © 2020 com.geetest. All rights reserved.
//

import Foundation

public class ListNode {
    public var val: Int
    public var next: ListNode?
    
    public init() {
        self.val = 0;
        self.next = nil;
    }
    
    public init(_ val: Int) {
        self.val = val;
        self.next = nil;
    }
    
    public init(_ val: Int, _ next: ListNode?) {
        self.val = val;
        self.next = next;
    }
}

class Homework_Week_01 {
    /**
     26. 删除排序数组中的重复项
     
     给定一个排序数组，你需要在 原地 删除重复出现的元素，使得每个元素只出现一次，返回移除后数组的新长度。

     不要使用额外的数组空间，你必须在 原地 修改输入数组 并在使用 O(1) 额外空间的条件下完成。

      

     示例 1:

     给定数组 nums = [1,1,2],

     函数应该返回新的长度 2, 并且原数组 nums 的前两个元素被修改为 1, 2。

     你不需要考虑数组中超出新长度后面的元素。
     示例 2:

     给定 nums = [0,0,1,1,1,2,2,3,3,4],

     函数应该返回新的长度 5, 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4。

     你不需要考虑数组中超出新长度后面的元素。
      

     说明:

     为什么返回数值是整数，但输出的答案是数组呢?

     请注意，输入数组是以「引用」方式传递的，这意味着在函数里修改输入数组对于调用者是可见的。

     你可以想象内部操作如下:

     // nums 是以“引用”方式传递的。也就是说，不对实参做任何拷贝
     int len = removeDuplicates(nums);

     // 在函数里修改输入数组对于调用者是可见的。
     // 根据你的函数返回的长度, 它会打印出数组中该长度范围内的所有元素。
     for (int i = 0; i < len; i++) {
         print(nums[i]);
     }

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        if nums.count > 1 {
            // count 记录重复元素的个数
            var i = 1, count = 0, len = nums.count
            while i < len {
                if nums[i] == nums[i - 1] {
                    count += 1
                } else {
                    nums[i - count] = nums[i]
                }
                i += 1
            }
            
            return len - count
        }
        
        return nums.count
    }
    
    /**
     21. 合并两个有序链表
     
     将两个升序链表合并为一个新的 升序 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

      

     示例：

     输入：1->2->4, 1->3->4
     输出：1->1->2->3->4->4

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/merge-two-sorted-lists
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        if nil == l1 || nil == l2 {
            return nil == l1 ? l2 : l1
        }
        
        let dummy = ListNode(0)
        var head: ListNode? = dummy
        var head1 = l1, head2 = l2
        while nil != head1 && nil != head2 {
            if head1!.val < head2!.val {
                head?.next = head1
                head = head1
                head1 = head1?.next
            } else {
                head?.next = head2
                head = head2
                head2 = head2?.next
            }
        }
        
        head?.next = nil == head1 ? head2 : head1
        
        return dummy.next
    }
    
    /**
     1. 两数之和
     
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
     88. 合并两个有序数组
     
     给你两个有序整数数组 nums1 和 nums2，请你将 nums2 合并到 nums1 中，使 nums1 成为一个有序数组。

      

     说明:

     初始化 nums1 和 nums2 的元素数量分别为 m 和 n 。
     你可以假设 nums1 有足够的空间（空间大小大于或等于 m + n）来保存 nums2 中的元素。
      

     示例:

     输入:
     nums1 = [1,2,3,0,0,0], m = 3
     nums2 = [2,5,6],       n = 3

     输出: [1,2,2,3,5,6]

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/merge-sorted-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var i = m - 1, j = n - 1, k = m + n - 1
        while i >= 0 && j >= 0 {
            if nums1[i] > nums2[j] {
                nums1[k] = nums1[i]
                k -= 1
                i -= 1
            } else {
                nums1[k] = nums2[j]
                k -= 1
                j -= 1
            }
        }
        
        while j >= 0 {
            nums1[k] = nums2[j]
            k -= 1
            j -= 1
        }
    }
    
    /**
     189. 旋转数组
     
     给定一个数组，将数组中的元素向右移动 k 个位置，其中 k 是非负数。

     示例 1:

     输入: [1,2,3,4,5,6,7] 和 k = 3
     输出: [5,6,7,1,2,3,4]
     解释:
     向右旋转 1 步: [7,1,2,3,4,5,6]
     向右旋转 2 步: [6,7,1,2,3,4,5]
     向右旋转 3 步: [5,6,7,1,2,3,4]
     示例 2:

     输入: [-1,-100,3,99] 和 k = 2
     输出: [3,99,-1,-100]
     解释:
     向右旋转 1 步: [99,-1,-100,3]
     向右旋转 2 步: [3,99,-1,-100]
     说明:

     尽可能想出更多的解决方案，至少有三种不同的方法可以解决这个问题。
     要求使用空间复杂度为 O(1) 的 原地 算法。

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/rotate-array
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func rotate(_ nums: inout [Int], _ k: Int) {
        var count = k
        if count > nums.count {
            count -= nums.count
        }
        if count > 0 && count < nums.count {
            reverse(&nums, 0, nums.count - 1)
            reverse(&nums, 0, count - 1)
            reverse(&nums, count, nums.count - 1)
        }
    }
    
    func reverse(_ nums: inout [Int], _ left: Int, _ right: Int) {
        var i = left, j = right
        while i < j {
            let temp = nums[i]
            nums[i] = nums[j]
            nums[j] = temp
            i += 1
            j -= 1
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
        if k <= 0 || arr.count <= 0 {
            return [Int]()
        }
        
        var res = [Int]()
        for val in arr {
            res.append(val)
        }
        
        var start = 0
        var end = res.count - 1
        var index = partition(&res, start, end)
        while index != k - 1 {
            if index > k - 1 {
                end = index - 1
                index = partition(&res, start, end)
            } else {
                start = index + 1
                index = partition(&res, start, end)
            }
        }
        
        return Array(res[0 ..< k])
    }
    
    private func partition(_ arr: inout [Int],  _ start: Int, _ end: Int) -> Int {
        if start >= 0 && end >= start && arr.count > start && arr.count > end {
            if start == end {
                return start
            }
            
            let random = start + (end - start)/2
            swap(&arr, random, end)
            
            var small = start - 1
            for index in start ..< end {
                if arr[index] < arr[end] {
                    small += 1
                    if index != small {
                        swap(&arr, small, index)
                    }
                }
            }
            
            small += 1
            swap(&arr, small, end)
            
            return small
        }
        
        return -1
    }
    
    private func swap(_ arr: inout [Int],  _ indexA: Int, _ indexB: Int) {
        if indexA >= 0 && indexB >= 0 && arr.count > indexA && arr.count > indexB {
            let temp = arr[indexA]
            arr[indexA] = arr[indexB]
            arr[indexB] = temp
        }
    }
    
    /**
     42. 接雨水
     
     给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。


     上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。 感谢 Marcos 贡献此图。

     示例:

     输入: [0,1,0,2,1,0,1,3,2,1,2,1]
     输出: 6

     来源：力扣（LeetCode）
     链接：https://leetcode-cn.com/problems/trapping-rain-water
     著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
     */
    func trap(_ height: [Int]) -> Int {
        // 方法1
        // 暴力法
        /*
        var sum = 0
        if height.count > 2 {
            for i in 1 ... height.count - 2 {
                // 求左边最高
                var leftMax = 0
                for j in 0 ... i - 1 {
                    leftMax = max(leftMax, height[j])
                }
                
                // 求右边最高
                var rightMax = 0
                for j in i + 1 ... height.count - 1 {
                    rightMax = max(rightMax, height[j])
                }
                
                if min(leftMax, rightMax) > height[i] {
                    sum += (min(leftMax, rightMax) - height[i])
                }
            }
        }
        return sum
        */
        
        // 方法2
        // 动态规划
        // 若 dp[i] 表示第 i 列左边最高的高度，则 dp[i] = max(dp[i - 1], height[i - 1])
        // 若 dp[i] 表示第 i 列右边最高的高度，则 dp[i] = max(dp[i + 1], height[i + 1])
        
        var sum = 0
        
        if height.count > 2 {
            var maxLeft = [Int](repeating: 0, count: height.count)
            for i in 1 ... height.count - 1 {
                maxLeft[i] = max(maxLeft[i - 1], height[i - 1])
            }
            
            var maxRight = [Int](repeating: 0, count: height.count)
            for i in (0 ... height.count - 2).reversed() {
                maxRight[i] = max(maxRight[i + 1], height[i + 1])
            }
            
            for i in 1 ... height.count - 2 {
                if min(maxLeft[i], maxRight[i]) > height[i] {
                    sum += (min(maxLeft[i], maxRight[i]) - height[i])
                }
            }
        }
        
        return sum
    }
}

/**
 641. 设计循环双端队列
 
 设计实现双端队列。
 你的实现需要支持以下操作：

 MyCircularDeque(k)：构造函数,双端队列的大小为k。
 insertFront()：将一个元素添加到双端队列头部。 如果操作成功返回 true。
 insertLast()：将一个元素添加到双端队列尾部。如果操作成功返回 true。
 deleteFront()：从双端队列头部删除一个元素。 如果操作成功返回 true。
 deleteLast()：从双端队列尾部删除一个元素。如果操作成功返回 true。
 getFront()：从双端队列头部获得一个元素。如果双端队列为空，返回 -1。
 getRear()：获得双端队列的最后一个元素。 如果双端队列为空，返回 -1。
 isEmpty()：检查双端队列是否为空。
 isFull()：检查双端队列是否满了。
 示例：

 MyCircularDeque circularDeque = new MycircularDeque(3); // 设置容量大小为3
 circularDeque.insertLast(1);                    // 返回 true
 circularDeque.insertLast(2);                    // 返回 true
 circularDeque.insertFront(3);                    // 返回 true
 circularDeque.insertFront(4);                    // 已经满了，返回 false
 circularDeque.getRear();                  // 返回 2
 circularDeque.isFull();                        // 返回 true
 circularDeque.deleteLast();                    // 返回 true
 circularDeque.insertFront(4);                    // 返回 true
 circularDeque.getFront();                // 返回 4
  
  

 提示：

 所有值的范围为 [1, 1000]
 操作次数的范围为 [1, 1000]
 请不要使用内置的双端队列库。

 来源：力扣（LeetCode）
 链接：https://leetcode-cn.com/problems/design-circular-deque
 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
 */
class MyCircularDeque {
    private var deque: [Int]
    private var k: Int
    
    /** Initialize your data structure here. Set the size of the deque to be k. */
    init(_ k: Int) {
        self.k = k
        self.deque = [Int]()
        self.deque.reserveCapacity(k)
    }
    
    /** Adds an item at the front of Deque. Return true if the operation is successful. */
    func insertFront(_ value: Int) -> Bool {
        if !self.isFull() {
            self.deque.insert(value, at: 0)
            return true
        }
        
        return false
    }
    
    /** Adds an item at the rear of Deque. Return true if the operation is successful. */
    func insertLast(_ value: Int) -> Bool {
        if !self.isFull() {
            self.deque.append(value)
            return true
        }
        
        return false
    }
    
    /** Deletes an item from the front of Deque. Return true if the operation is successful. */
    func deleteFront() -> Bool {
        if !self.isEmpty() {
            self.deque.removeFirst()
            return true
        }
        
        return false
    }
    
    /** Deletes an item from the rear of Deque. Return true if the operation is successful. */
    func deleteLast() -> Bool {
        if !self.isEmpty() {
            self.deque.removeLast()
            return true
        }
        
        return false
    }
    
    /** Get the front item from the deque. */
    func getFront() -> Int {
        if !self.isEmpty() {
            return self.deque.first!
        }
        
        return -1
    }
    
    /** Get the last item from the deque. */
    func getRear() -> Int {
        if !self.isEmpty() {
            return self.deque.last!
        }
        
        return -1
    }
    
    /** Checks whether the circular deque is empty or not. */
    func isEmpty() -> Bool {
        return self.deque.isEmpty
    }
    
    /** Checks whether the circular deque is full or not. */
    func isFull() -> Bool {
        return self.k == self.deque.count
    }
}
