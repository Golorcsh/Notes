# Leetcode

## 排序

### 快速排序

```c_cpp
int partition(vector<int> &arr, int left, int right) {
    swap(arr[left], arr[right]);//首尾交换，将基数放在最后
    int i = left - 1;
    for (int j = left; j < right; j++) {
        if (arr[j] < arr[right]) {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    i++;//当前小于基数的最后一位，需要位移一位
    swap(arr[i], arr[right]);//将基数放回排序后的位置
    return i;
}
void quickSort(vector<int> &arr, int left, int right) {
    if (left >= right)
        return;
    int pivot = partition(arr, left, right);
    quickSort(arr, left, pivot - 1);
    quickSort(arr, pivot + 1, right);
}
```

### 堆排序（大根）

```c_cpp
void maxHeap(vector<int> &arr, int i, int heapSize) {//堆排序 小根堆只要修改为取最小值即可
    int l = i * 2 + 1, r = i * 2 + 2, largest = i;
    if (l < heapSize && arr[l] > arr[largets]) {
        largest = l;
    }
    if (r < heapSize && arr[r] > arr[largets]) {
        largest = r;
    }
    if (largest != i) {
        swap(arr[i], arr[largest]);
        maxHeap(arr, largest, heapSize);
    }
}

void buildHeap(vector<int> &arr, int heapSize) {//建堆
    int n = heapSize / 2;
    for (int i = n; i >= 0; i--) {
        maxHeap(arr, i, heapSize);
    }
}
```

### 归并排序

```c_cpp
void merge(vector<int> &nums, vector<int> &temp, int left, int mid, int right) {
    //将nums[left..right]拷贝到temp[left..right]中辅助
    copy(nums.begin() + left, nums.begin() + right + 1, temp.begin() + left);
    //两个区间的起点
    int i = left, j = mid + 1;
    for (int k = left; k <= right; k++) {
        if (i == mid + 1) {//左区间已经使用完，右值归并
            nums[k] = temp[j++];
        } else if (j == right + 1) {//右区间已经使用完，左值归并
            nums[k] = temp[i++];
        } else if (temp[i] <= temp[j]) {//右区间已经使用完或temp[i]<=temp[j]，左值归并
            nums[k] = temp[i++];
        } else {//temp[i]>temp[j],右值归并
            nums[k] = temp[j++];
        }
    }
}
//归并排序
void mergerSort(vector<int> &nums, int left, int right) {
    //区间只有一个元素
    if (left >= right)
        return;
    //划分递归区间
    int mid = (right - left) / 2 + left;
    mergerSort(nums, left, mid);
    mergerSort(nums, mid + 1, right);
    //优化:当左边有序和右边有序，且左边的右边界<=右边的左边界,说明 nums[left..right]一整个已经是有序，不需要合并；
    if (nums[mid] <= nums[mid + 1])
        return;
    //归并
    vector<int> temp(nums.size());
    merge(nums, temp, left, mid, right);
}
```

## 一 、二叉树专题

### 二叉树

#### Morris遍历

中序遍历的一种方式，占用常数空间，其思想是利用空闲的指针。

```c_cpp
//找到前继节点
TreeNode *getSuccessor(TreeNode *node) {
  TreeNode *pre = node->left;
  while (pre->right != nullptr && pre->right != node) {
    pre = pre->right;
  }
  return pre;
}
TreeNode *morris_travel(TreeNode *root) {
  TreeNode *node = root;
  while (node != nullptr) {
    if (node->left == nullptr) {
      cout << node->val << " ";
      node = node->right;
    } else {
      //找到前继节点（即中序遍历时的前一个位置，当前的节点的左子树，然后一直右子树）
      TreeNode *succ = getSuccessor(node);
      //若为空则使用右节点执行父节点
      if (succ->right == nullptr) {
        succ->right = node;
        node = node->left;
      } else {
        succ->right = nullptr;
        cout << node->val << " ";
        node = node->right;
      }
    }
  }
}
```

#### [144. 二叉树的前序遍历](https://leetcode-cn.com/problems/binary-tree-preorder-traversal/)

```c_cpp
class Solution {
    public:
    //迭代
    vector<int> preorderTraversal(TreeNode *root) {
    stack<TreeNode *> st;
    vector<int> ret;
    while (root != nullptr || !st.empty()) {
      //不断往左走，并且将根节点存入栈中，直到遇到空
      while (nullptr != root) {
        ret.push_back(root->val);
        st.push(root);
        root = root->left;
      }
      //取出父节点，往父节点的右子树走
      root = st.top();
      st.pop();
      root = root->right;
    }
    return ret;
  }
    //递归
    vector<int> preorderTraversal(TreeNode *root) {
        vector<int> ret;
        DFS(root, ret);
        return ret;
    }
    void DFS(TreeNode *root, vector<int> &ret) {
        if (root == nullptr)
            return;
        ret.push_back(root->val);
        DFS(root->left, ret);
        DFS(root->right, ret);
    }
};
```

#### [145. 二叉树的后序遍历](https://leetcode-cn.com/problems/binary-tree-postorder-traversal/)

```c_cpp
//迭代
vector<int> postorderTraversal(TreeNode *root) {
    stack<TreeNode *> st;
    vector<int> ret;
    TreeNode *prev = nullptr;//用来识别当前节点的前继是否是自己的右子树
    while (root != nullptr  || !st.empty()) {
        //不断往左走，并且将根节点存入栈中，直到遇到空
        while (nullptr != root) {
            st.push(root);
            root = root->left;
        }
        //取出父节点，往父节点的右子树走
        root = st.top();
        st.pop();
        //若当前节点的右子树为空，或当前结点的前继是自己的右子树(说明已经遍历过）
        if (root->right == nullptr || root->right == prev) {
            ret.push_back(root->val);
            prev = root;
            root = nullptr;
        } else {
            st.push(root);
            root = root->right;
        }
    }
    return ret;
}
//递归
vector<int> postorderTraversal1(TreeNode *root) {
    vector<int> ret;
    DFS(root, ret);
    return ret;
}
void DFS(TreeNode *root, vector<int> &ret) {
    if (root == nullptr)
        return;
    DFS(root->left, ret);
    DFS(root->right, ret);
    ret.push_back(root->val);
}
```

#### [94. 二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```c_cpp
//迭代：模拟电脑栈
vector<int> inorderTraversal(TreeNode *root) {
    stack<TreeNode *> st;
    vector<int> ret;
    while (root != nullptr || !st.empty()) {
        //不断往左走，直到遇到空节点
        while (root != nullptr) {
            st.push(root);
            root = root->left;
        }
        //取出父节点
        root = st.top();
        st.pop();
        ret.push_back(root->val);
        //往右走
        root = root->right;
    }
    return ret;
}
//递归
vector<int> inorderTraversal1(TreeNode *root) {
    vector<int> ret;
    DFS(root, ret);
    return ret;
}
void DFS(TreeNode *root, vector<int> &ret) {
    if (root == nullptr)
        return;
    DFS(root->left, ret);
    ret.push_back(root->val);
    DFS(root->right, ret);
}
```

#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```c_cpp
//思路：广度遍历，在取出的时候将一层都取出
vector<vector<int>> levelOrder(TreeNode *root) {
    if (root == nullptr)
        return {};
    vector<vector<int>> ret;
    queue<TreeNode *> queue;
    queue.push(root);
    while (!queue.empty()) {
        vector<int> temp;
        //将一层都取出
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            TreeNode *p = queue.front();
            queue.pop();
            temp.push_back(p->val);
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        ret.push_back(temp);
    }
    return ret;
}
```

#### [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

```c_cpp
//思路：广度遍历，然后修改方向
vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
    if (root == nullptr)
        return {};
    int direct = false;
    queue<TreeNode *> queue;
    queue.push(root);
    vector<vector<int>> ret;
    while (!queue.empty()) {
        vector<int> temp;
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            TreeNode *p = queue.front();
            queue.pop();
            temp.push_back(p->val);
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        //修改方向
        if (direct) {
            std::reverse(temp.begin(), temp.end());
        }
        direct = ~direct;
        ret.push_back(temp);
    }
    return ret;
}
```

### 二叉搜索树

#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```c_cpp
//迭代中序遍历
bool isValidBST(TreeNode *root) {
    stack<TreeNode*>st;
    long prev = LONG_MIN;
    while(root!= nullptr||!st.empty()){
        while(root!= nullptr){
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();
        if(root->val<=prev){
            return false;
        }
        prev = root->val;
        root=root->right;
    }
    return true;
}

bool isValidBST(TreeNode *root) {
    return DFS(root);
}
//中序遍历
long pre = LONG_MIN;
bool DFS(TreeNode *root) {
    if (root == nullptr)
        return true;
    //左子树
    bool left = DFS(root->left);
    //当前节点小于前继节点（中序遍历的前继节点值）
    if (root->val <= pre)
        return false;
    //更新前继节点
    pre = root->val;
    //右子树
    bool right = DFS(root->right);
    return left && right;
}
```

#### [99. 恢复二叉搜索树](https://leetcode-cn.com/problems/recover-binary-search-tree/)

```c_cpp
//迭代中序遍历
void recoverTree(TreeNode *root) {
    stack<TreeNode *> st;
    TreeNode *pre = nullptr;
    TreeNode *x = nullptr;
    TreeNode *y = nullptr;
    while (root != nullptr || !st.empty()) {
        while (nullptr != root) {
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();
        //判断前继节点是否大于当前节点，如果是则记录
        if (pre != nullptr && pre->val > root->val) {
            y = root;
            if (x == nullptr)
                x = pre;
        }
        //更新前继节点
        pre = root;
        root = root->right;
    }
    swap(x->val, y->val);
}
```

#### [230. 二叉搜索树中第K小的元素](https://leetcode-cn.com/problems/kth-smallest-element-in-a-bst/)

```c_cpp
//迭代中序遍历
int kthSmallest1(TreeNode *root, int k) {
    stack<TreeNode *> st;
    int i = 0;
    while (root != nullptr || !st.empty()) {
        while (root != nullptr) {
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();
        //判断当前遍历到第几个
        if (++i == k)
            return root->val;
        root = root->right;
    }
    return 0;
}
//DFS 中序遍历
int kthSmallest(TreeNode *root, int k) {
    int ret = 0;
    DFS(root, k, ret);
    return ret;
}
void DFS(TreeNode *root, int &k, int &ret) {
    if (root == nullptr)
        return;
    DFS(root->left, k, ret);
    if (--k == 0)
        ret = root->val;
    if (k > 0)
        DFS(root->right, k, ret);
}
```

#### [285. 二叉搜索树中的中序后继](https://leetcode-cn.com/problems/inorder-successor-in-bst)

```c_cpp
题目：给一个二叉查找树(什么是二叉查找树)，以及一个节点，求该节点的中序遍历后继，如果没有返回null
输入: {1,#,2}, p = 1
输出: 2
解释:
  1
   \\\\
    2
//迭代中序遍历
TreeNode *inorderSuccessor(TreeNode *root, TreeNode *p) {
    stack<TreeNode *> st;
    TreeNode *prev = nullptr;
    while (root != nullptr || !st.empty()) {
        while (root != nullptr) {
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();
        //当前继节点等于目标节点时，返回当前节点(即目标节点的后继节点)
        if (prev == p)
            return root;
        prev = root;
        root = root->right;
    }
    return nullptr;
}
```

#### [272. 最接近的二叉搜索树值 II](https://leetcode-cn.com/problems/closest-binary-search-tree-value-ii)

```c_cpp
题目：给定一个不为空的二叉搜索树和一个目标值 target，请在该二叉搜索树中找到最接近目标值 target 的 k 个值。
注意：
给定的目标值 target 是一个浮点数
你可以默认 k 值永远是有效的，即 k ≤ 总结点数
题目保证该二叉搜索树中只会存在一种 k 个值集合最接近目标值
示例：
输入: root = [4,2,5,1,3]，目标值 = 3.714286，且 k = 2
    4
   / \\\\
  2   5
 / \\\\
1   3
输出: [4,3]
struct cmp {
    //差值大的在上
    bool operator()(pair<float, int> &x, pair<float, int> &y) {
        return x.first < y.first;
    }
};
//思路使用优先队列(最大堆),存储K个最近值，以差值作为比较
vector<int> closestKValues(TreeNode *root, double target, int k) {
    stack<TreeNode *> st;
    priority_queue<pair<float, int>, vector<pair<float, int>>, cmp> priority_queue;
    while (root || !st.empty()) {
        while (root) {
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();
        //用最大堆处理
        if (priority_queue.size() < k)//不足K个，直接插入
            priority_queue.push(make_pair(fabs(root->val - target), root->val));
        //当有K个后，且当前的差值更小，则弹出堆顶，压入堆
        else if (priority_queue.size() == k && priority_queue.top().first > fabs(root->val - target)) {
            priority_queue.pop();
            priority_queue.push(make_pair(fabs(root->val - target), root->val));
        }
        //当队列大小等于K且当前的差值大于堆顶，则说明没有更小的差值了，直接break
        if (priority_queue.size() == k && priority_queue.top().first <= root->val - target)
            break;
        root = root->right;
    }
    vector<int> ret;
    while (!priority_queue.empty()) {
        ret.push_back(priority_queue.top().second);
        priority_queue.pop();
    }
    return ret;
}
```

### 二叉树混合题

#### [257. 二叉树的所有路径](https://leetcode-cn.com/problems/binary-tree-paths/)

```c_cpp
Tag:遍历 回溯
vector<string> binaryTreePaths(TreeNode *root) {
    vector<string> ret;
    string path{};
    DFS(root, path, ret);
    return ret;
}
//使用pass by value 就可以不用回溯，若想使用回溯则使用vector<int> 最后在处理变成字符串
void DFS(TreeNode *root, string path, vector<string> &paths) {
    if (root) {
        path += to_string(root->val);
        if (root->left == nullptr && root->right == nullptr) {
            paths.push_back(path);
        } else {
            path += "->";
            DFS(root->left, path, paths);
            DFS(root->right, path, paths);
        }
    }
}
```

#### [270. 最接近的二叉搜索树值](https://leetcode-cn.com/problems/closest-binary-search-tree-value)

```c_cpp
//迭代中序遍历 记录差值
int closestValue(TreeNode *root, double target) {
    stack<TreeNode *> st;
    double difference = numeric_limits<double>::max();
    int ret = 0;
    while (root != nullptr || !st.empty()) {
        while (root != nullptr) {
            st.push(root);
            root = root->left;
        }
        root = st.top();
        st.pop();

        if (fabs(root->val - target) < difference) {
            difference = fabs(root->val - target);
            ret = root->val;
        }
        root = root->right;
    }
    return ret;
}
```

#### [199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

```c_cpp
//BFS
vector<int> rightSideView1(TreeNode *root) {
    if (root == nullptr)
        return {};
    queue<TreeNode *> queue;
    queue.push(root);
    vector<int> ret;
    while (!queue.empty()) {
        int size = queue.size();
        TreeNode *p = nullptr;
        for (int i = 0; i < size; i++) {
            p = queue.front();
            queue.pop();
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        //当遍历完后，p指向当前层的最后一个node,直接插入
        ret.push_back(p->val);
    }
    return ret;
}

vector<int> rightSideView(TreeNode *root) {
    vector<int> res;
    DFS(root, 0, res);
    return res;
}
void DFS(TreeNode *root, int depth, vector<int> &res) {
    if (root == nullptr)
        return;
    // 如果当前节点所在深度还没有出现在res里，说明在该深度下当前节点是第一个被访问的节点，因此将当前节点加入res中。
    if (depth == res.size()) {
        res.push_back(root->val);
    }
    depth++;
    DFS(root->right, depth, res);
    DFS(root->left, depth, res);
}
```

#### [337. 打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

```c_cpp
int rob(TreeNode *root) {
    auto sub = DFS(root);
    return max(sub.rob, sub.not_rob);
}
struct subTree {
    int rob;
    int not_rob;
};
subTree DFS(TreeNode *root) {
    if (root == nullptr)
        return {0, 0};
    subTree left = DFS(root->left);
    subTree right = DFS(root->right);
    //如果抢当前节点，则无法抢子节点
    int rob = root->val + left.not_rob + right.not_rob;
    //如果不抢当前节点，则可以抢子节点,也可以不抢子节点，取最大值
    int not_rob = max(left.rob, left.not_rob) + max(right.rob, right.not_rob);
    return {rob, not_rob};
}
```

#### [255. 验证前序遍历序列二叉搜索树](https://leetcode-cn.com/problems/verify-preorder-sequence-in-binary-search-tree)

```c_cpp
//栈迭代
bool verifyPreorder(vector<int> &preorder) {
    int size = preorder.size();
    if (size == 0)
        return true;
    stack<int> st;
    st.push(preorder[0]);//根节点入栈；
    int temp = INT_MIN;
    //遇到左子树时，全部入栈，遇到右子树时，将与其平级的左子树出栈【它具有大于平级左子树的性质】
    //出现出栈的时候，新来的元素必定是大于已经出栈的元素。
    for (int i = 1; i < size; i++) {
        if (preorder[i] < temp)
            return false;
        //当出现一个值大于栈顶，说明出现一个右子树,然后不停的出栈左子树，直到遇到比当前值大的数
        while (!st.empty() && preorder[i] > st.top()) {
            temp = st.top();
            st.pop();
        }
        //左子树入栈
        st.push(preorder[i]);
    }
    return true;
}
```

#### [114. 二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

```c_cpp
//递归
void flatten(TreeNode *root) {
    /*
     * 思路
     * 交换当前节点的左右子树
     * 然后找到当前节点最右边的一个节点，然后将当前节点的左子树挂在最后一个节点上
     * 在将当前节点的左子树置空
     * 然后递归当前节点的右子树
     */
    if (root == nullptr)
        return;
    swap(root->left, root->right);
    TreeNode *cur = root;
    while (cur->right)
        cur = cur->right;
    cur->right = root->left;
    root->left = nullptr;
    flatten(root->right);
}
//先序遍历存储结点，然后在处理
void flatten1(TreeNode *root) {
    vector<TreeNode *> nodes;
    DFS(root, nodes);
    int size = nodes.size();
    for (int i = 1; i < size; i++) {
        nodes[i - 1]->right = nodes[i];
        nodes[i - 1]->left = nullptr;
    }
}
void DFS(TreeNode *root, vector<TreeNode *> &nodes) {
    if (root == nullptr)
        return;
    nodes.push_back(root);
    DFS(root->left, nodes);
    DFS(root->right, nodes);
}
```

#### [156. 上下翻转二叉树](https://leetcode-cn.com/problems/binary-tree-upside-down)

```c_cpp
TreeNode *upsideDownBinaryTree(TreeNode *root) {
    if(root = nullptr)
        return root;
    return DFS(root);
}
TreeNode * DFS(TreeNode*root){
    if(root->left==nullptr&&root->right==nullptr)//叶子结点
        return root;
    TreeNode * true_root = DFS(root->left);//不断往左子树走，直到叶子结点
    root->left->right = root->right;//将当前root的右子树，设置为左子树的右子树
    root->left->left = root;//将当前节点设置为左子树的左子树
    root->left = nullptr;//端口当前节点的左右子树
    root->right = nullptr;
    return true_root;//返回真正的root节点
}
```

#### [116. 填充每个节点的下一个右侧节点指针](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node/)

```c_cpp
//利用父节点的next指针
Node *connect(Node *root) {
    if (root == nullptr)
        return nullptr;
    Node *most_left = root;
    while (most_left->left != nullptr) {
        Node *head = most_left;
        while (head != nullptr) {
            //连接同一个父节点的子节点(处于同一个父节点）
            head->left->next = head->right;
            //若干父节点还有兄弟节点，则当前的右节点连接到兄弟节点的左节点，(不在同一个父节点）
            if (head->next != nullptr)
                head->right->next = head->next->left;
            head = head->next;
        }
        most_left = most_left->left;//更新下一层的最左节点
    }
    return root;
}
//BFS
Node *connect1(Node *root) {
    if (root == nullptr || !root->left && !root->right)
        return root;
    queue<Node *> queue;
    queue.push(root);
    while (!queue.empty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            Node *p = queue.front();
            queue.pop();
            //当到最后一个时，设置为空
            p->next = i == size - 1 ? nullptr : queue.front();
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
    }
    return root;
}
```

#### [117. 填充每个节点的下一个右侧节点指针 II](https://leetcode-cn.com/problems/populating-next-right-pointers-in-each-node-ii/)

```c_cpp
Node *connect(Node *root) {
    //使用哑结点作为每一层的开始
    Node *dummy = new Node(0);
    //使用指向哑结点的值来遍历每一层
    Node *dummy_cur = dummy;
    Node *cur = root;
    while (cur != nullptr) {
        while (cur != nullptr) {
            if (cur->left) {
                dummy_cur->next = cur->left;
                dummy_cur = dummy_cur->next;
            }
            if (cur->right) {
                dummy_cur->next = cur->right;
                dummy_cur = dummy_cur->next;
            }
            cur = cur->next;
        }
        //更新到下一层
        cur = dummy->next;
        //将哑的next指针置空,以便下次使用
        dummy->next = nullptr;
        //重置到哑结点
        dummy_cur = dummy;
    }
    return root;
}
//BFS
Node *connect1(Node *root) {
    if (root == nullptr)
        return root;
    queue<Node *> queue;
    queue.push(root);
    while (!queue.empty()) {
        int size = queue.size();
        for (int i = 0; i < size; i++) {
            Node *p = queue.front();
            queue.pop();
            p->next = i == size - 1 ? nullptr : queue.front();
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
    }
    return root;
}
```

#### [297. 二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

```c_cpp
class Codec {
    public:
    // Encodes a tree to a single string.
    string serialize(TreeNode *root) {
        if (root == nullptr)
            return {};
        string res;
        my_serialize(root, res);
        return res;
    }
    //DFS 先序遍历
    void my_serialize(TreeNode *node, string &str) {
        if (node == nullptr) {//空节点使用#代替
            str += "# ";
            return;
        }
        str += to_string(node->val) + " ";
        my_serialize(node->left, str);
        my_serialize(node->right, str);
    }
    // Decodes your encoded data to tree.
    TreeNode *deserialize(string data) {
        if (data.empty())
            return nullptr;
        istringstream ss(data);
        return my_deserialize(ss);
    }
    //DFS 先序遍历, 使用istringstream 分割字符
    TreeNode *my_deserialize(istringstream &ss) {
        string temp;
        ss >> temp;
        if (temp == "#")
            return nullptr;
        auto *root = new TreeNode(stoi(temp));
        root->left = my_deserialize(ss);
        root->right = my_deserialize(ss);
        return root;
    }
};
```

### 二叉树递归

#### [95. 不同的二叉搜索树 II](https://leetcode-cn.com/problems/unique-binary-search-trees-ii/)

```c_cpp
vector<TreeNode *> generateTrees(int n) {
    if (n ==c 0)
        return {nullptr};
    return generateTree(1, n);
}
vector<TreeNode *> generateTree(int start, int end) {
    if (start > end)
        return {nullptr};
    vector<TreeNode *> all_tree;
    //枚举所有根节点
    for (int i = start; i <= end; i++) {
        //获得所有可能的左子树
        vector<TreeNode *> lefts = generateTree(start, i - 1);
        //获得所有可能的右子树
        vector<TreeNode *> rights = generateTree(i + 1, end);
        //生成根节点，并从左子树合集中选一个作为左子树，在右子树合集中选一个右子树，最后添加到返回值中
        for (const auto &left: lefts) {
            for (const auto &right: rights) {
                auto *cur = new TreeNode(i);
                cur->left = left;
                cur->right = right;
                //构建好的树，添加到返回值，以供上层使用
                all_tree.push_back(cur);
            }
        }
    }
    //将构建好的树的根节点集合返回上层使用
    return all_tree;
}
```

#### [96. 不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```c_cpp
//递归（超时）
int numTrees(int n) {
    //递归出口，当n==0 一种空树，当n==1一种树
    if(n==0||n==1)
        return 1;
    int res=0;
    //以i为根节点遍历，累积所有可能
    for (int i = 1; i <=n; i++)
        res+= numTrees(i-1)* numTrees(n-i);
    return res;
}
//dp
int numTrees1(int n) {
    vector<int> dp(n + 1, 0);
    //初始化边界
    dp[0] = 1;//0个节点，只有1种树(空树）
    dp[1] = 1;//1个节点。只有一种树
    //从2个节点开始
    for (int i = 2; i <= n; i++) {
        for (int j = 1; j <= i; j++) {
            //以i为根节点，左子树和右子树数量变化，所有可能
            dp[i] += dp[j - 1] * dp[i - j];
        }
    }
    return dp[n];
}
```

#### [124. 二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```c_cpp
  int maxRes = INT_MIN;
  int maxPathSum(TreeNode *root) {
    postOrder(root);
    return maxRes;
  }
  //DFS 后序遍历，需要先处理子问题，然后才能处理本节点
  int postOrder(TreeNode *root) {
    if (root == nullptr)
      return 0;
    //若若左右子树为负数则舍弃
    int left = max(postOrder(root->left), 0);
    int right = max(postOrder(root->right), 0);
    //更新路径最大值,路径的最大值可以包含root节点
    maxRes = max(maxRes, root->val + left + right);
    //只选择最大的一侧，才能形成路径，若选择两边则不符合题意
    return root->val + max(left, right);
  }
```

#### [235. 二叉搜索树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```c_cpp
TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    TreeNode *ancestor = root;
    while (true) {
        if (p->val < ancestor->val && q->val < ancestor->val) {//两个节点都小于root,说明在root左子树
            ancestor = ancestor->left;
        } else if (p->val > ancestor->val && q->val > ancestor->val) {//两个节点都大于root,说明在root右子树
            ancestor = ancestor->right;
        } else {//当都不符合说明抵达分叉点,直接退出
            break;
        }
    }
    return ancestor;
}
```

#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```c_cpp
TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
    if (root == nullptr)
        return nullptr;
    if (root->val == p->val || root->val == q->val)//若当前节点是其中一个节点那么他就是最近公告祖先
        return root;
    auto left = lowestCommonAncestor(root->left, p, q);//左子树
    auto right = lowestCommonAncestor(root->right, p, q);//右子树

    if (left != nullptr && right != nullptr)//若左右子树都找到了，说明当前就是最近公告节点
        return root;
    if (left == nullptr)//若左子树找不到，那么只能在右子树
        return right;
    if (right == nullptr)//若右子树找不到，那么只能在左子树
        return left;
    return nullptr;
}
```

#### [250. 统计同值子树](https://leetcode-cn.com/problems/count-univalue-subtrees)

```c_cpp
int countUnivalSubtrees(TreeNode* root) {
    int ret=0;
    helper(root,ret);
    return ret;
}
//先看左右子树是否是同值子树，然后在对当前节点做判断
bool helper(TreeNode*root,int &ret){
    if(root==nullptr)
        return true;
    auto left=helper(root->left,ret);
    auto right=helper(root->left,ret);
    if(left&&right){
        if(root->left&& root->left->val!=root->val)
            return false;
        if(root->right&& root->right->val!=root->val)
            return false;
        ret++;
        return true;
    }
    return false;
}
```

#### [100. 相同的树](https://leetcode-cn.com/problems/same-tree/)

```c_cpp
bool isSameTree(TreeNode *p, TreeNode *q) {
    return DFS(p, q);
}

bool DFS(TreeNode *node1, TreeNode *node2) {
    if (node1 == nullptr && node2 == nullptr)//两个节点都为空
        return true;
    else if (node1 == nullptr || node2 == nullptr)//有一方节点为空
        return false;
    else if (node1->val != node2->val)//值不相等
        return false;
    else
        return DFS(node1->left, node2->left) && DFS(node1->right, node2->right);//当前节点相等，看子节点
}
```

#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```c_cpp
int maxDepth(TreeNode *root) {
    if (root) {
        int left = maxDepth(root->left);
        int right = maxDepth(root->right);
        return max(left, right) + 1;
    }
    return 0;
}
```

#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```c_cpp
TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
    unordered_map<int, int> map;
    int size = inorder.size();
    //记录中序遍历的值与对应的用于加速寻找root的位置
    for (int i = 0; i < size; ++i) {
        map[inorder[i]] = i;
    }
    return DFS(preorder, inorder, 0, size - 1, 0, size - 1, map);

}
/**
   *
   * @param preorder
   * @param inorder
   * @param pLeft  前序遍历的左边界
   * @param pRight 前序遍历的右边界
   * @param iLeft   中序遍历的右边界
   * @param iRight  中序遍历的右边界
   * @param map    记录下标
   * @return
   */
TreeNode *DFS(vector<int> &preorder, vector<int> &inorder, int pLeft, int pRight, int iLeft, int iRight,
              unordered_map<int, int> &map) {
    if (pLeft > pRight || iLeft > iRight)
        return nullptr;
    TreeNode *root = new TreeNode(preorder[pLeft]);
    int rootIndex = map[preorder[pLeft]];
    //通过rootIndex,计算左右子树有多少节点，从而设置前序遍历的范围
    root->left = DFS(preorder, inorder, pLeft + 1, rootIndex - iLeft + pLeft, iLeft, rootIndex - 1, map);
    root->right = DFS(preorder, inorder, rootIndex - iLeft + pLeft + 1, pRight, rootIndex + 1, iRight, map);
    return root;
}
```

#### [106. 从中序与后序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)

```c_cpp
//方法和105一样，注意post便利的范围计算
TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
    int size = inorder.size();
    unordered_map<int, int> map;
    for (int i = 0; i < size; ++i) {
        map[inorder[i]] = i;
    }
    return DFS(inorder, postorder, 0, size - 1, 0, size - 1, map);
}
TreeNode *DFS(vector<int> &inorder, vector<int> &postorder,
              int iLeft, int iRight, int pLeft, int pRight,
              unordered_map<int, int> &map) {
    if (iLeft > iRight || pLeft > pRight)
        return nullptr;
    auto root = new TreeNode(postorder[pRight]);
    int rootIndex = map[postorder[pRight]];
    int inorderLeftSize = rootIndex - iLeft;
    root->left = DFS(inorder, postorder, iLeft, rootIndex - 1, pLeft, rootIndex - iLeft + pLeft - 1, map);
    root->right = DFS(inorder, postorder, rootIndex + 1, iRight, rootIndex - iLeft + pLeft, pRight - 1, map);
    return root;
}
```

#### [108. 将有序数组转换为二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-array-to-binary-search-tree/)

```c_cpp
TreeNode *sortedArrayToBST(vector<int> &nums) {
    if (nums.empty())
        return nullptr;
    return DFS(nums, 0, nums.size() - 1);
}
TreeNode *DFS(vector<int> &nums, int left, int right) {
    if (left > right)
        return nullptr;
    int mid = (right - left) / 2 + left;//取中间值若为根节点
    auto root = new TreeNode(nums[mid]);
    root->left = DFS(nums, left, mid - 1);//对[left..mid-1]递归作为左子树
    root->right = DFS(nums, mid + 1, right);//对[mid+1..right]递归作为右子树
    return root;
}
```

#### [109. 有序链表转换二叉搜索树](https://leetcode-cn.com/problems/convert-sorted-list-to-binary-search-tree/)

```c_cpp
ListNode *getMidNode(ListNode *left, ListNode *right) {
    ListNode *slow = left;
    ListNode *fast = left;
    while (fast != right && fast->next != right) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}
TreeNode *buildTree(ListNode *left, ListNode *right) {
    //设当前链表的左端点为 left，右端点 right，包含关系为「左闭右开」，即left 包含在链表中而 right 不包含在链表中。
    //为什么要设定「左闭右开」的关系？由于题目中给定的链表为单向链表，访问后继元素十分容易，但无法直接访问前驱元素。
    // 因此在找出链表的中位数节点 mid 之后，如果设定「左闭右开」的关系，我们就可以直接用 (left,mid) 以及 ((mid.next,right) 来表示左右子树对应的列表了。
    // 并且，初始的列表也可以用 (head,null) 方便地进行表示，其中 null 表示空节点。
    if (left == right)
        return nullptr;
    auto midNode = getMidNode(left, right);
    auto root = new TreeNode(midNode->val);
    root->left = buildTree(left, midNode);
    root->right = buildTree(midNode->next, right);
    return root;
}
TreeNode *sortedListToBST(ListNode *head) {
    return buildTree(head, nullptr);
}
//使用数组空间换时间
TreeNode *sortedListToBST1(ListNode *head) {
    vector<int> values;
    while (head) {
        values.push_back(head->val);
        head = head->next;
    }
    return DFS(values, 0, values.size() - 1);
}
TreeNode *DFS(vector<int> &value, int left, int right) {
    if (left > right)
        return nullptr;
    int mid = (right - left) / 2 + left;
    auto root = new TreeNode(value[mid]);
    root->left = DFS(value, left, mid - 1);
    root->right = DFS(value, mid + 1, right);
    return root;
}
```

#### [110. 平衡二叉树](https://leetcode-cn.com/problems/balanced-binary-tree/)

```c_cpp
bool isBalanced(TreeNode *root) {
    return root ? DFS(root) > 0 : true;
}
int DFS(TreeNode *root) {
    if (root == nullptr)
        return 0;
    int left = DFS(root->left);//左子树深度
    int right = DFS(root->right);//右子树深度
    if (left < 0 || right < 0 || abs(left - right) > 1)//左子树或右子树不以及不平衡或这当前不平衡则返回-1
        return -1;
    return max(left, right) + 1;//返回当前高度
}
```

#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)'

```cpp
int minDepth(TreeNode *root) {
    if (root == nullptr)
        return 0;
    queue<pair<TreeNode *, int>> queue;//节点以及节点所在的层次
    queue.push({root, 1});
    while (!queue.empty()) {
        auto *p = queue.front().first;
        auto depth = queue.front().second;
        queue.pop();
        if (p->left == nullptr && p->right == nullptr)//找到第一个叶子节点直接返回
            return depth;
        if (p->left)
            queue.push(make_pair(p->left, depth + 1));
        if (p->right)
            queue.push(make_pair(p->right, depth + 1));
    }
    return 0;
}
int minDepth1(TreeNode *root) {
    return DFS(root);
}
int DFS(TreeNode *root) {
    if (root == nullptr)//节点返回0
        return 0;
    if (root->left == nullptr && root->right == nullptr)//叶子节点返回1
        return 1;
    int minDepth = INT_MAX;
    if (root->left) {//计算左子树
        minDepth = min(DFS(root->right), minDepth);
    }
    if (root->right) {//计算右子树
        minDepth = min(DFS(root->right), minDepth);
    }
    return minDepth + 1;
}
```

#### [112. 路径总和](https://leetcode-cn.com/problems/path-sum/)

```cpp
bool hasPathSum(TreeNode *root, int targetSum) {
    return DFS(root, targetSum, 0);
  }
bool DFS(TreeNode *root, int &target, int sum) {
  if (root == nullptr)
    return false;
  sum += root->val;//记录和
  if (sum == target && (root->left == nullptr && root->right == nullptr))//判断是否等于target,并且为叶子节点
    return true;
  return DFS(root->left, target, sum) || DFS(root->right, target, sum);//左右子树任意一个为true
}
```

#### [113. 路径总和 II](https://leetcode-cn.com/problems/path-sum-ii/)

```c_cpp
vector<vector<int>> res;
vector<int> combination;
vector<vector<int>> pathSum(TreeNode *root, int targetSum) {
  backTrack(root, targetSum);
  return res;
}
void backTrack(TreeNode *root, int target) {
  if (root == nullptr)
    return;
  combination.push_back(root->val);
  target -= root->val;
  if (target == 0 && (root->left == nullptr && root->right == nullptr)) {//不能提前返回，不然无法弹出节点，到下面的时候都会返回为空
    res.push_back(combination);
  }
  backTrack(root->left, target);
  backTrack(root->right, target);
  combination.pop_back();
}
```

#### [129. 求根节点到叶节点数字之和](https://leetcode-cn.com/problems/sum-root-to-leaf-numbers/)

```cpp
int sumNumbers(TreeNode *root) {
  int sum = 0;
  string str{};
  DFS(root, str, sum);
  return sum;
}
//先处理root，在处理左右子树
void DFS(TreeNode *root, string &str, int &sum) {
  if (root == nullptr)
    return;
  str += to_string(root->val);
  //到叶子节点则计算累加值
  if (root->left == nullptr && root->right == nullptr)
    sum += stoi(str);
  DFS(root->left, str, sum);
  DFS(root->right, str, sum);
  //走完叶子节点，在走叶子节点的左右后回溯
  str.pop_back();
}
```

#### [222. 完全二叉树的节点个数](https://leetcode-cn.com/problems/count-complete-tree-nodes/)

```cpp
int countNodes(TreeNode *root) {
  if (root == nullptr)
    return 0;
  int level = 0;
  //记录层数
  TreeNode *node = root;
  while (node->left != nullptr) {
    level++;
    node = node->left;
  }
  //最有一层的第一个编号2^h 后最后一个(2^h+1)-1
  int left = 1 << level, right = (1 << (level + 1)) - 1;
  //二分
  while (left < right) {
    int mid = (right - left + 1) / 2 + left;
    if (exist(root, level, mid)) {
      left = mid;
    } else {
      right = mid - 1;
    }
  }
  return left;
}
bool exist(TreeNode *root, int level, int k) {
  //每个节点的编号位数为h+1
  int bits = 1 << (level - 1);//去除二进制的第一个位数
  TreeNode *node = root;
  while (node != nullptr && bits > 0) {
    if (!(bits & k)) {
      node = node->left;
    } else
      node = node->right;
    bits >>= 1;
  }
  return node != nullptr;
}

int countNodes1(TreeNode *root) {
  return DFS(root);
}
int DFS(TreeNode *root) {
  if (root == nullptr)
    return 0;
  int left = DFS(root->left);//左子树的节点数量
  int right = DFS(root->right);//右子树的节点数量
  return left + right + 1;//左右子树和本身
}
```

#### [226. 翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```cpp
TreeNode *invertTree(TreeNode *root) {
  DFS(root);
  return root;
}
void DFS(TreeNode *root) {
  if (root == nullptr)
    return;
  /*先交换自己的左右子树*/
  swap(root->left, root->right);
  /*递归地处理左子树和右子树*/
  DFS(root->left);
  DFS(root->right);
}
```

#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```cpp
bool isSymmetric1(TreeNode *root) {
  return DFS(root, root);
}
bool DFS(TreeNode *node1, TreeNode *node2) {
  if (node1 == nullptr && node2 == nullptr) {
    return true;
  }
  /*任意一个为空或值value不相等，职业返回false*/
  if (node1 == nullptr || node2 == nullptr) {
    return false;
  } else if (node1->val != node2->val) {
    return false;
  }
  /*镜像的检查子树*/
  return DFS(node1->left, node2->right) && DFS(node1->right, node2->left);
}
/*迭代*/
bool isSymmetric(TreeNode *root) {
  queue<TreeNode *> queue;
  queue.push(root);
  queue.push(root);
  while (!queue.empty()) {
    auto u = queue.front();
    queue.pop();
    auto v = queue.front();
    queue.pop();
    if (u == nullptr && v == nullptr)
      continue;
    if ((u == nullptr || v == nullptr) || v->val != u->val)
      return false;
    queue.push(u->left);
    queue.push(v->right);

    queue.push(u->right);
    queue.push(v->left);
  }
  return true;
}
```

## 二 、BFS / DFS / TOPO + 回溯

###  回溯

#### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

```clike
bool isMatch(string s, string p) {
  int m = s.size();
  int n = p.size();
  vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
  /*第一个空字符对空字符为true*/
  dp[0][0] = true;
  /*当遇到*号时，看两个位置之前字符是否匹配，若匹配的话可以是的星号(作用为0)不出现*/
  for (int i = 2; i <= n; i += 2) {
    /*eg: a*b.ac */
    /*当遇到*号时，看前两个字符是否匹配，若匹配的话可以是的*号不出现*/
    dp[0][i] = dp[0][i - 2] && p[i - 1] == '*';/*第一行与空字符匹配，考虑第二个字符时候为*号能否匹配*/
  }

  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (p[j - 1] == '*') {/*星号两种情况*/
        /*第一种情况是：使得星号不出现（即空字符），则看前面是否匹配*/
        /*第二种情况是：使得星号重复前一个字符，则需要p前一个字符和s当前是否匹配,且之前是可以匹配*/
        dp[i][j] = dp[i][j - 2] || ((s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
      } else {
        /*若当前字符相等，或则p[i]为点可以匹配，则看前一个字符是否匹配*/
        dp[i][j] = (p[j - 1] == '.' || s[i - 1] == p[j - 1]) && dp[i - 1][j - 1];
      }
    }
  }
  return dp[m][n];
}
/*递归*/
bool isMatch(string s, string p) {
  int len1 = s.size();
  int len2 = p.size();
  /*出口*/
  if (len2 == 0)
    return len1 == 0;
  /*判断第一个字符是否可以匹配*/
  bool firstMatch = (len1 > 0 && len2 > 0 && (p[0] == s[0] || p[0] == '.'));
  /*第二个字符为*两种情况*/
  if (len2 > 1 && p[1] == '*') {
    /*第一种情况为让*取空，让p移动第二个字符,s不移动*/
    /*第二种情况为让*再取前面的字符，则p不用移动，而让s移动,前提第一个要匹配(不然没有意义)*/
    return isMatch(s, p.substr(2)) || (firstMatch && isMatch(s.substr(1), p));
  } else
    /*第二个字符不为*则进入递归判断后面的字符*/
    return firstMatch && isMatch(s.substr(1), p.substr(1));
}
```

####  [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```cpp
unordered_map<char, string> map{
    {'2', "abc"},
    {'3', "def"},
    {'4', "ghi"},
    {'5', "jkl"},
    {'6', "mno"},
    {'7', "pqrs"},
    {'8', "tuv"},
    {'9', "wxyz"}
};
vector<string> letterCombinations(string digits) {
  if (digits.empty())
    return {};
  vector<string> res;
  string combination{};
  backTrack(digits, 0, combination, res);
  return res;
}
/*回溯*/
void backTrack(string &digits, int index, string &combination, vector<string> &res) {
  if (index == digits.size()) {
    res.push_back(combination);
    return;
  }
  /*获得按钮映射的字符串*/
  char num = digits[index];
  string digit = map[num];
  /*遍历字符穿*/
  for (const auto &ch: digit) {
    /*记录字符*/
    combination.push_back(ch);
    backTrack(digits, index + 1, combination, res);
    /*回溯*/
    combination.pop_back();
  }
}
```

#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```clike
vector<string> generateParenthesis(int n) {
  if (n == 0)
    return {};
  string buckets = {};
  vector<string> res;
  backTrack(n, 0, 0, buckets, res);
  return res;
}

void backTrack(int n, int l, int r, string &buckets, vector<string> &res) {
  if (buckets.size() == 2 * n) {
    res.push_back(buckets);
    return;
  }
  if (l < n) {
    buckets.push_back('(');
    backTrack(n, l + 1, r, buckets, res);
    buckets.pop_back();
  }
  /*右括号必须小于等于左括号，才能合法*/
  if (r < l) {
    buckets.push_back(')');
    backTrack(n, l, r + 1, buckets, res);
    buckets.pop_back();
  }
}
```

#### [37. 解数独](https://leetcode-cn.com/problems/sudoku-solver/)

```clike
/*回溯，使用空间换时间*/
bool rows[9][9];/*9行，每行九个元素*/
bool cols[9][9];/*9列，每列9个元素*/
bool block[3][3][9];/*公用3*3个block每个block有9个元素*/
vector<pair<int, int>> space;/*需要填数字的位置*/
bool valid;
void solveSudoku(vector<vector<char>> &board) {
  memset(rows, false, sizeof(rows));
  memset(cols, false, sizeof(cols));
  memset(block, false, sizeof(block));
  valid = false;
  for (int i = 0; i < 9; ++i) {
    for (int j = 0; j < 9; ++j) {
      if (board[i][j] == '.') {
        space.emplace_back(i, j);
      } else {
        int digit = board[i][j] - '0' - 1;
        rows[i][digit] = cols[j][digit] = block[i / 3][j / 3][digit] = true;/*标记为使用过*/
      }
    }
  }
  DFS(board, 0);
}
void DFS(vector<vector<char>> &board, int pos) {
  if (pos == space.size()) {/*所有位置以及填完*/
    valid = true;
    return;
  }
  auto[i, j]  =space[pos];/*取出需要填写的坐标*/
  /*从1-9依次填入,并判断是以及填写完毕*/
  for (int digit = 0; digit < 9 && !valid; ++digit) {
    /*判断digit是否在行、列，块中出现过，出现了直接跳过*/
    if (rows[i][digit] || cols[j][digit] || block[i / 3][j / 3][digit]) {
      continue;
    }
    rows[i][digit] = cols[j][digit] = block[i / 3][j / 3][digit] = true;/*若没有使用个，则使用，并标记已使用*/
    board[i][j] = digit + 1 + '0';/*填写*/
    DFS(board, pos + 1);/*进入下一个填写位置*/
    rows[i][digit] = cols[j][digit] = block[i / 3][j / 3][digit] = false;/*回溯,取消使用标记*/
  }
}

/*暴力法,可以通过，速度慢*/
void solveSudoku(vector<vector<char>> &board) {
  DFS(board, 0, 0);
}
bool DFS(vector<vector<char>> &board, int i, int j) {
  if (i == 9)return true;/*所有行都填完*/
  if (j == 9)return DFS(board, i + 1, 0);/*当前行填完，判断下一行*/

  if (board[i][j] != '.')/*当前已有数字，进入下一个数字*/
    return DFS(board, i, j + 1);
  for (int k = '1'; k <= '9'; ++k) {
    if (isValid(board, i, j, k) == false) continue;//判断当前位置填入k是否合法，不合法则换下一个
    board[i][j] = k;/*填入k*/
    if (DFS(board, i, j + 1) == true) {/*进入一下个位置填数,若后面成功则返回true*/
      return true;
    }
    board[i][j] = '.';/*回溯*/
  }
  return false;
}
bool isValid(vector<vector<char>> &board, int i, int j, char k) {
  /*判断所在的行、列和格子中是否以及出现过*/
  for (int row = 0; row < 9; ++row) {
    if (board[row][j] == k) return false;
  }
  for (int col = 0; col < 9; ++col) {
    if (board[i][col] == k) return false;
  }
  /*计算所在格子的左上角位置*/
  int x = i / 3 * 3;
  int y = j / 3 * 3;
  for (int p = 0; p < 3; ++p) {
    for (int q = 0; q < 3; ++q) {
      if (board[x + p][y + q] == k) return false;
    }
  }
  return true;
}
```

####  [39. 组合总和](https://leetcode-cn.com/problems/combination-sum/)

```clike
vector<int> combination;
vector<vector<int>> res;
vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
  backTrack(candidates, target, 0, 0);
  return res;
}
void backTrack(vector<int> &candidates, const int &target, int sum, int index) {
  if (sum == target) {/*能够组合*/
    res.push_back(combination);
    return;
  }
  if (sum > target)/*超出*/
    return;

  for (int i = index; i < candidates.size(); ++i) {
    combination.push_back(candidates[i]);
    /*每个数字可以无限选，不用+1*/
    backTrack(candidates, target, sum + candidates[i], i);
    combination.pop_back();
  }
}
```

#### [40. 组合总和 II](https://leetcode-cn.com/problems/combination-sum-ii/)

```clike
vector<int> combination;
vector<vector<int>> res;
vector<vector<int>> combinationSum2(vector<int> &candidates, int target) {
  std::sort(candidates.begin(), candidates.end());
  backTrack(candidates, 0, 0, target);
  return res;
}
void backTrack(vector<int> &candidates, int index, int sum, const int target) {
  if (sum == target) {
    res.push_back(combination);
    return;
  }
  if (sum > target)
    return;

  for (int i = index; i < candidates.size(); ++i) {
    /*若当前数字选用在则跳过*/
    if (i > index && candidates[i] == candidates[i - 1])
      continue;
    combination.push_back(candidates[i]);
    /*i+1不重复选取*/
    backTrack(candidates, i + 1, sum + candidates[i], target);
    combination.pop_back();
  }
}
```

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

```c_cpp
bool isMatch(string s, string p) {
    int len1 = s.size();
    int len2 = p.size();
    vector<vector<bool>> dp(len1 + 1, vector<bool>(len2 + 1, false));
    dp[0][0] = true;
    /*初始化边界条件，dp[i][0]与空匹配，不匹配恒等于false（第一列）*/
    /*初始化边界条件，dp[0][j]空字符匹配，则看从开始有多少个连续的*号，则有多少个true*/
    for (int j = 1; j <= len2; ++j) {
        if (p[j - 1] == '*')
            dp[0][j] = true;
        else
            break;
    }
    for (int i = 1; i <= len1; ++i) {
        for (int j = 1; j <= len2; ++j) {
            if (p[j - 1] == '*') {/*当前字符*有两种情况*/
                /*情况一：使用*匹配空字符，则看i和j-1是否匹配*/
                /*情况一：使用*匹配，则看i-1和j是否匹配*/
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if (s[i - 1] == p[j - 1] || p[j - 1] == '?') {/*当前字符可以匹配*/
                dp[i][j] = dp[i - 1][j - 1];/*则看前面的是否为true*/
            }
        }
    }
    return dp[len1][len2];
}
bool isMatch(string s, string p) {
    int len1 = s.size();
    int len2 = p.size();
    /*
     * dp[i][j] 代表
     * [i...end]
     * [j...end] 是否匹配
     */
    vector<vector<bool>> dp(len1 + 1, vector<bool>(len2 + 1));
    /*从后往前*/
    for (int i = len1; i >= 0; --i) {
        for (int j = len2; j >= 0; --j) {
            /*初始化边界*/
            if (i == len1 && j == len2) {
                dp[i][j] = true;
                continue;
            }
            bool firstMatch = i != len1 && j != len2 && (s[i] == p[j] || p[j] == '?' || p[j] == '*');
            if (j != len2 && p[j] == '*') {/*当前字符为*号则有两种情况*/
                /*情况一：将其匹配空，则看i和j+1是否匹配*/
                /*情况二：将其匹i+1后面的字符，则看i+1和j是否匹配,且当前第一个需要匹配*/
                dp[i][j] = dp[i][j + 1] || (firstMatch && dp[i + 1][j]);
            } else {
                dp[i][j] = firstMatch && dp[i + 1][j + 1];/*当前字符匹配，则看后面的的是否匹配*/
            }
        }
    }
    return dp[0][0];
}
```

#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

```c_cpp
vector<vector<int>> res;
vector<int> combination;
vector<vector<int>> permute(vector<int> &nums) {
    unordered_set<int> set;
    bacTrack(nums, set);
    return res;
}
void bacTrack(vector<int> &nums, unordered_set<int> &set) {
    /*出口*/
    if (combination.size() == nums.size()) {
        res.push_back(combination);
        return;
    }
    /*遍历nums*/
    for (const auto &n : nums) {
        /*没有使用个的数字在进入下一个*/
        if (!set.count(n)) {
            /*记录*/
            combination.push_back(n);
            set.insert(n);
            bacTrack(nums, set);
            /*回溯*/
            set.erase(n);
            combination.pop_back();
        }
    }
}
```

#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

```c_cpp
vector<int> combination;
vector<vector<int>> res;
vector<vector<int>> permuteUnique(vector<int> &nums) {
    vector<bool> used(nums.size(), false);
    /*将相同的数字排到一起，以便过滤*/
    std::sort(nums.begin(), nums.end());
    backTrack(nums, used);
    return res;
}
void backTrack(vector<int> &nums, vector<bool> &used) {
    if (combination.size() == nums.size()) {
        res.push_back(combination);
        return;
    }
    int size = nums.size();
    for (int i = 0; i < size; ++i) {
        /*当前数字已使用过*/
        if (used[i])
            continue;
        /*过滤相同的数字，若当前和前一个相同，由于回溯，前一个没有使用，则需要过滤*/
        if (i > 0 && nums[i - 1] == nums[i] && !used[i - 1])
            continue;
        used[i] = true;
        combination.push_back(nums[i]);
        backTrack(nums, used);
        combination.pop_back();
        used[i] = false;
    }
}
```

#### [51. N 皇后](https://leetcode-cn.com/problems/n-queens/)

```c_cpp
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> res;
    vector<int> queens(n, -1);
    unordered_set<int> columns;/*列*/
    unordered_set<int> diagonals1;/*斜下  行下标-列下标 的差相等*/
    unordered_set<int> diagonals2;/*斜上  行下标+列下标 的差相等*/
    /*不用记录row的位置，每一次递归都是王下一层，不会重复*/
    bacTrack(res, queens, n, 0, columns, diagonals1, diagonals2);
    return res;
}
void bacTrack(vector<vector<string>> &res, vector<int> &queens, int n, int row,
              unordered_set<int> &columns, unordered_set<int> &diagonals1, unordered_set<int> &diagonals2) {
    if (row == n) {
        /*到最后一行表示以及填完,记录答案*/
        res.push_back(generateBoard(queens, n));
    } else {
        for (int i = 0; i < n; ++i) {
            /*判断当前列是否已有皇后*/
            if (columns.find(i) != columns.end()) 
                continue;
            /*判断斜下位置是由已有皇后*/
            int diagonal1 = row - i;/*行-列的差相等的为在一条斜下的路径上*/
            if (diagonals1.find(diagonal1) != diagonals1.end()) 
                continue;
            /*判断斜下位置是由已有皇后*/
            int diagonal2 = row + i;/*行+列的和相等的为在一条斜下的路径上*/
            if (diagonals2.find(diagonal2) != diagonals2.end())
                continue;
            queens[row] = i;/*皇后位置 row行，i列*/
            /*列、斜下，斜上方向记录*/
            columns.insert(i);
            diagonals1.insert(diagonal1);
            diagonals2.insert(diagonal2);
            /*递归*/
            bacTrack(res, queens, n, row + 1, columns, diagonals1, diagonals2);
            /*回溯*/
            queens[row] = -1;
            columns.erase(i);
            diagonals1.erase(diagonal1);
            diagonals2.erase(diagonal2);
        }
    }
}
vector<string> generateBoard(vector<int> &queens, int n) {
    /*根据皇后位置生产填写皇后的位置图*/
    auto board = vector<string>();
    for (int i = 0; i < n; ++i) {
        string row = string(n, '.');
        row[queens[i]] = 'Q';
        board.push_back(row);
    }
    return board;
}
```

#### [52. N皇后 II](https://leetcode-cn.com/problems/n-queens-ii/)

```c_cpp
int totalNQueens(int n) {
    int res = 0;
    unordered_set<int> columns;/*记录列*/
    unordered_set<int> diagonals1;/*记录斜下*/
    unordered_set<int> diagonals2;/*记录斜上*/
    backTrack(res, 0, n, columns, diagonals1, diagonals2);
    return res;
}
void backTrack(int &res, int row, int n,
               unordered_set<int> &columns, unordered_set<int> &diagonals1, unordered_set<int> &diagonals2) {
    if (row == n) {
        res++;
        return;
    }

    for (int i = 0; i < n; ++i) {
        /*判断列和对角线是否存在皇后(由于每一次都是往下一次，因此不必检查行)*/
        if (columns.find(i) != columns.end())
            continue;
        int diagonal1 = row - i;
        if (diagonals1.find(diagonal1) != diagonals1.end())
            continue;
        int diagonal2 = row + i;
        if (diagonals2.find(diagonal2) != diagonals2.end())
            continue;

        /*记录*/
        columns.insert(i);
        diagonals1.insert(diagonal1);
        diagonals2.insert(diagonal2);
        backTrack(res, row + 1, n, columns, diagonals1, diagonals2);
        /*回溯*/
        columns.erase(i);
        diagonals1.erase(diagonal1);
        diagonals2.erase(diagonal2);
    }
}
```

#### [60. 排列序列](https://leetcode-cn.com/problems/permutation-sequence/)

```c_cpp
string getPermutation(int n, int k) {
    /*思路：由于n个数组成的序列全排列有n!种*/
    /*则已 a1 开头的全排列有 (n-1)！种*/
    vector<bool> used(n + 1);
    string res{};
    backTrack(used, res, n, k);
    return res;
}
void backTrack(vector<bool> &used, string &res, int n, int k) {
    int size = res.size();
    if (size == n) {/*出口*/
        return;
    }
    int remain_fac = factorial(n - size - 1);/*计算已使用的和当前使用后，与剩下的字符能组成的全排列个数*/
    for (int i = 1; i <= n; ++i) {
        if (used[i])
            continue;
        if (remain_fac < k) {/*组成的全排列数量小于K，说明第K个排列不在当前字符开头的递归的子树中，跳过*/
            k -= remain_fac;
            continue;
        }
        used[i] = true;
        res += static_cast<char>(i + '0');
        backTrack(used, res, n, k);
        /*不需要回溯，应为直接计算到叶子节点*/
    }
}
/*求n的阶乘*/
int factorial(int n) {
    int res = 1;
    while (n > 0) {
        res *= n;
        n--;
    }
    return res;
}
```

#### [77. 组合](https://leetcode-cn.com/problems/combinations/)

```c_cpp
vector<int> combination;
vector<vector<int>> res;
vector<vector<int>> combine(int n, int k) {
    if (k <= 0 || n < k)
        return {};
    bacTrack(n, k, 1);
    return res;
}
void bacTrack(int n, int k, int index) {
    /*剪枝:当前所选的数量+当前区间可以选的数<k，则无法构成*/
    if (combination.size() + (n - index + 1) < k)
        return;
    if (combination.size() == k) {
        res.push_back(combination);
        return;
    }

    for (int i = index; i <= n; ++i) {
        combination.push_back(i);
        bacTrack(n, k, i + 1);/*从剩下的数中选*/
        combination.pop_back();
    }
}
```

#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

```c_cpp
vector<vector<int>> subsets(vector<int> &nums) {
    vector<vector<int>> res;
    res.push_back(vector<int>());/*空集*/
    for (int i = 0; i < nums.size(); ++i) {
        /*遍历现有的所有子集，然后填充生成新的子集，然后在插入到答案中*/
        int size = res.size();
        for (int j = 0; j < size; ++j) {
            auto cur = res[j];
            cur.push_back(nums[i]);
            res.push_back(cur);
        }
    }
    return res;
}
vector<int> combination;
vector<vector<int>> res;
vector<vector<int>> subsets1(vector<int> &nums) {
    backTrack(nums, 0);
    return res;
}
void backTrack(vector<int> &nums, int index) {
    /*记录所有的子集*/
    res.push_back(combination);
    int size = nums.size();
    for (int i = index; i < size; ++i) {
        combination.push_back(nums[i]);
        backTrack(nums, i + 1);
        combination.pop_back();
    }
}
```

#### [79. 单词搜索](https://leetcode-cn.com/problems/word-search/)

```c_cpp
int direct[4][2] = {{1, 0}, {-1, 0},
                    {0, 1}, {0, -1}};
bool exist(vector<vector<char>> &board, string word) {
    int m = board.size(), n = board[0].size();
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            /*每个点尝试作为开始*/
            if (backTrack(board, word, 0, i, j))
                return true;
        }
    }
    return false;
}
bool backTrack(vector<vector<char>> &board, const string &word, int index, int x, int y) {
    int m = board.size(), n = board[0].size();
    if (word[index] != board[x][y])
        return false;
    /*匹配到最后一个字符*/
    if (index == word.size() - 1)
        return true;
    /*标记已使用*/
    char original = board[x][y];
    board[x][y] = '.';
    for (int i = 0; i < 4; ++i) {
        int nx = direct[i][0] + x, ny = direct[i][1] + y;
        if (nx >= 0 && nx < m && ny >= 0 && ny < n && board[nx][ny] != '.') {/*剪枝：越界或已使用*/
            if (backTrack(board, word, index + 1, nx, ny))
                return true;
        }
    }
    /*回溯*/
    board[x][y] = original;
    return false;
}
```

#### [90. 子集 II](https://leetcode-cn.com/problems/subsets-ii/)

```c_cpp
vector<int> combination;
vector<vector<int>> res;
vector<bool> used;
vector<vector<int>> subsetsWithDup(vector<int> &nums) {
    std::sort(nums.begin(), nums.end());
    used.resize(nums.size());
    backTrack(nums, 0);
    return res;
}

void backTrack(vector<int> &nums, int index) {
    res.push_back(combination);

    int size = nums.size();
    for (int i = index; i < size; ++i) {
        /*去重*/
        if (i > index && !used[i - 1] && nums[i] == nums[i - 1])
            continue;
        used[i] = true;
        combination.push_back(nums[i]);
        backTrack(nums, i + 1);
        used[i] = false;
        combination.pop_back();
    }
}
```

#### [93. 复原 IP 地址](https://leetcode-cn.com/problems/restore-ip-addresses/)

```c_cpp
int SET_COUNT = 4;
vector<string> ans;
vector<int> segments;
vector<string> restoreIpAddresses(string s) {
    segments.resize(SET_COUNT);
    backTrack(s, 0, 0);
    return ans;
}

void backTrack(const string &s, int segId, int segStart) {
    if (segId == SET_COUNT) {/*第四端*/
        if (segStart == s.size()) {/*且已经使用完字符串*/
            string ip;
            for (int i = 0; i < SET_COUNT; ++i) {
                ip += to_string(segments[i]);
                if (i != SET_COUNT - 1)
                    ip += '.';
            }
            ans.push_back(move(ip));
        }
        return;
    }
    if (segStart == s.size())/*还未找到第四段，而字符使用完毕*/
        return;
    if (s[segStart] == '0') {/*不能有前导0，如果当前是0,当前段ip只能为0*/
        segments[segId] = 0;
        backTrack(s, segId + 1, segStart + 1);
    }
    /*遍历每一种可能*/
    int addr = 0;
    for (int i = segStart; i < s.size(); ++i) {
        addr = addr * 10 + (s[i] - '0');
        if (addr > 0 && addr <= 0xff) {
            segments[segId] = addr;
            backTrack(s, segId + 1, i + 1);
        } else
            break;
    }
}
```

#### [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

```c_cpp
/*思路：首先广度优先遍历，将可以通过改变一个字符转换的相连，构建成一个图*/
/*然后在深度遍历（回溯）找到最短路径*/
/*细节，从一个词拓展另一个词的使用，通过对其中一个字符从a-z修改，然后判断是否出现在dict中*/
vector<vector<string>> findLadders(string beginWord, string endWord, vector<string> &wordList) {
    vector<vector<string>> res;
    /*需要快速判断拓展的词是否出现，使用hashtable加速*/
    unordered_set<string> dict(wordList.begin(), wordList.end());
    /*特殊用例：没有结尾单词，没有结果*/
    if (dict.find(endWord) == dict.end())
        return res;
    /*从beginWord开始拓展，因此dict中不能包含有，不然有重复*/
    dict.erase(beginWord);

    /*广度有限遍历，构建图*/
    /*了避免记录不需要的边，我们需要记录扩展出的单词是在第几次扩展的时候得到的，key：单词，value：在广度优先遍历的第几层*/
    /*steps记录了当前已访问过的word集合和在第几层访问过*/
    unordered_map<string, int> steps;
    /*记录单词是从哪些单词拓展而来，是一对多*/
    unordered_map<string, set<string>> from;
    bool found = bfs(beginWord, endWord, dict, steps, from);
    if (found) {
        vector<string> path;
        path.push_back(endWord);
        dfs(from, path, beginWord, endWord, res);
    }
    return res;
}
bool bfs(string beginWord,
         string endWord,
         unordered_set<string> &dict,
         unordered_map<string, int> &steps,
         unordered_map<string, set<string>> &from) {
    int wordLen = beginWord.size();
    int step = 0;
    bool found = false;
    queue<string> queue;
    queue.push(beginWord);
    while (!queue.empty()) {
        step++;
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            string curWord = std::move(queue.front());
            queue.pop();
            string nextWord = curWord;
            for (int j = 0; j < wordLen; ++j) {
                const char origin = nextWord[j];
                for (int c = 'a'; c < 'z'; ++c) {
                    nextWord[j] = c;
                    /*如果拓展词和和本身同一层则不记录，只有不同才记录*/
                    if (steps.count(nextWord) && steps.at(nextWord) == step) {
                        from[nextWord].insert(curWord);
                    }
                    if (dict.find(nextWord) == dict.end()) {
                        continue;
                    }
                    dict.erase(nextWord);
                    queue.push(nextWord);
                    from[nextWord].insert(curWord);
                    steps[nextWord] = step;
                    if (nextWord == endWord) {/*由于有多条路径可以达到endWord,因此不能直接推出，将其标记即可*/
                        found = true;
                    }
                }
                nextWord[j] = origin;
            }
        }
        if (found) {/*以及达到endWord,不用继续构图*/
            break;
        }
    }
    return found;
}
void dfs(unordered_map<string, set<string>> &from,
         vector<string> &path,
         const string &beginWord,
         const string &cur,
         vector<vector<string>> &res) {
    /*到过来寻找*/
    if (cur == beginWord) {
        res.emplace_back(path.rbegin(), path.rend());/*将结果反转*/
        return;
    }
    /*遍历可以抵达cur的节点*/
    for (const auto &precursor : from.at(cur)) {
        path.push_back(precursor);
        dfs(from, path, beginWord, precursor, res);
        path.pop_back();
    }
}
```

#### [131. 分割回文串](https://leetcode-cn.com/problems/palindrome-partitioning/)

```c_cpp
/*DP预处理+回溯*/
vector<string> path;
vector<vector<string>> res;
vector<vector<bool>> dp;
vector<vector<string>> partition(string s) {
    /*使用dp预处理:dp[i,j]代表 字符串s[i,j]是否为回文串*/
    int size = s.size();
    dp.assign(size, vector<bool>(size, true));
    for (int i = size - 1; i >= 0; --i) {
        for (int j = i + 1; j < size; ++j) {
            /*i和j字符相等则看里面一层是否相等*/
            dp[i][j] = (s[i] == s[j]) && dp[i + 1][j - 1];
        }
    }
    backTrack(s, 0);
    return res;
}
void backTrack(string &s, int i) {
    int size = s.size();
    if (i == size) {
        res.push_back(path);
        return;
    }
    for (int j = i; j < size; ++j) {
        if (dp[i][j]) {
            path.push_back(s.substr(i, j - i + 1));
            backTrack(s, j + 1);
            path.pop_back();
        }
    }
}
/*回溯*/
vector<string> path;
vector<vector<string>> res;
vector<vector<string>> partition(string s) {
    backTrack(s, 0);
    return res;
}
void backTrack(string &s, int startIndex) {
    int size = s.size();
    if (startIndex == size) {
        res.push_back(path);
        return;
    }

    for (int i = startIndex; i < size; ++i) {
        if (isPalindrome(s, startIndex, i)) {/*判断是否是回文串*/
            path.push_back(s.substr(startIndex, i - startIndex + 1));/*截取[startIndex,i]*/
            backTrack(s, i + 1);
            path.pop_back();/*回溯*/
        }
    }
}
bool isPalindrome(string &s, int left, int right) {
    while (left < right) {
        if (s[left] != s[right])
            return false;
        left++;
        right--;
    }
    return true;
}

```

#### [140. 单词拆分 II](https://leetcode-cn.com/problems/word-break-ii/)

```c_cpp
unordered_set<string> wordSet;
unordered_map<int, vector<string>> ans;
vector<string> wordBreak(string s, vector<string> &wordDict) {
    wordSet = unordered_set<string>(wordDict.begin(), wordDict.end());
    backTrack(s, 0);
    return ans[0];

}
void backTrack(const string &s, int index) {
    if (!ans.count(index)) {

        if (index == s.size()) {
            ans[index] = {""};
            return;
        }

        ans[index] = {};
        for (int i = index + 1; i <= s.size(); ++i) {
            string word = s.substr(index, i - index);
            if (wordSet.count(word)) {
                backTrack(s, i);
                /*看已i开头的是否能分成单词,记忆话搜索*/
                for (auto &succ : ans[i]) {
                    /*如果后面的是可以分为单词的话，即不为空，则word+" "+succ*/
                    ans[index].push_back(succ.empty() ? word : word + " " + succ);
                }
            }
        }
    }
}

```

#### [212. 单词搜索 II](https://leetcode-cn.com/problems/word-search-ii/)

```c_cpp
/*前缀树+回溯*/
/*
思路：在构建好的前缀树中，使用回溯方法判断是否存在该单词。
*/
struct TrieNode {
    string word;
    unordered_map<char, TrieNode *> children;
    TrieNode() : word("") {
    }
};
void insertTrieNode(TrieNode *root, const string &word) {
    TrieNode *node = root;
    for (const auto &ch : word) {
        if (!node->children.count(ch)) {
            node->children[ch] = new TrieNode();
        }
        node = node->children[ch];
    }
    node->word = word;/*只用终点才有值，其余节点都问空，以便深度有限遍历是使用*/
}
int direct[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
bool backTrack(vector<vector<char>> &board, int x, int y, TrieNode *root, unordered_set<string> &res) {
    char ch = board[x][y];
    if (!root->children.count(ch))
        return false;
    root = root->children[ch];
    /*抵达单词*/
    if (!root->word.empty()) {
        res.insert(root->word);
    }
    board[x][y] = '.';
    for (auto &i : direct) {
        int nx = x + i[0], ny = y + i[1];
        if (nx >= 0 && nx < board.size() && ny >= 0 && ny < board[0].size() && board[nx][ny] != '.') {/*剪枝*/
            backTrack(board, nx, ny, root, res);
        }
    }
    board[x][y] = ch;
    return true;
}
vector<string> findWords(vector<vector<char>> &board, vector<string> &words) {
    TrieNode *root = new TrieNode();
    unordered_set<string> res;
    vector<string> ans;
    /*构建前缀树*/
    for (const auto &item : words) {
        insertTrieNode(root, item);
    }
    /*判断每个字符是否出现在前缀树中，若出现则进入寻找*/
    for (int i = 0; i < board.size(); ++i) {
        for (int j = 0; j < board[0].size(); ++j) {
            backTrack(board, i, j, root, res);
        }
    }
    ans.assign(res.begin(), res.end());
    return ans;
}
```

#### [216. 组合总和 III](https://leetcode-cn.com/problems/combination-sum-iii/)

```c_cpp
vector<vector<int>> res;
vector<int> combination;
vector<vector<int>> combinationSum3(int k, int n) {
    backTrack(k, n, 1, 0);
    return res;
}
void backTrack(const int k, const int n, int index, int sum) {
    if (sum == n && combination.size() == k) {
        res.push_back(combination);
        return;
    }
    for (int i = index; i <= 9; ++i) {
        if (sum + i > n)/*剪枝*/
            continue;
        combination.push_back(i);
        backTrack(k, n, i + 1, sum + i);
        combination.pop_back();
    }
}
```

### BFS

#### [101. 对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```c_cpp
bool isSymmetric(TreeNode *root) {
    return DFS(root, root);
    return BFS(root, root);
}

bool DFS(TreeNode *node1, TreeNode *node2) {
    if (node1 == nullptr && node2 == nullptr)
        return true;
    if (node1 == nullptr || node2 == nullptr)
        return false;
    if (node1->val != node2->val)
        return false;
    return DFS(node1->left, node2->right) && DFS(node1->right, node2->left);
}
bool BFS(TreeNode *node1, TreeNode *node2) {
    queue<TreeNode *> queue;
    queue.push(node1);
    queue.push(node2);
    while (!queue.empty()) {
        auto p = queue.front();
        queue.pop();
        auto q = queue.front();
        queue.pop();
        if (p == nullptr && q == nullptr)continue;
        if (p == nullptr || q == nullptr)return false;
        else if (p->val != q->val) return false;
        queue.push(q->left);
        queue.push(p->right);
        queue.push(q->right);
        queue.push(p->left);
    }
    return true;
}
```

#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```c_cpp
vector<vector<int>> levelOrder(TreeNode *root) {
    vector<vector<int>> res;
    if (root == nullptr)
        return res;
    queue<TreeNode *> queue;
    queue.push(root);
    while (!queue.empty()) {
        /*核心：将完整的一层取出来*/
        int size = queue.size();
        vector<int> temp;
        for (int i = 0; i < size; ++i) {
            auto p = queue.front();
            queue.pop();
            temp.push_back(p->val);
            if (p->left) {
                queue.push(p->left);
            }
            if (p->right) {
                queue.push(p->right);
            }
        }
        res.push_back(std::move(temp));
    }
    return res;
}
```

#### [103. 二叉树的锯齿形层序遍历](https://leetcode-cn.com/problems/binary-tree-zigzag-level-order-traversal/)

```c_cpp
vector<vector<int>> zigzagLevelOrder(TreeNode *root) {
    if (root == nullptr)
        return {};
    int direct = false;
    queue<TreeNode *> queue;
    queue.push(root);
    vector<vector<int>> res;
    while (!queue.empty()) {
        int size = queue.size();
        vector<int> temp(size);
        for (int i = 0; i < size; ++i) {
            auto p = queue.front();
            queue.pop();
            temp[i] = p->val;
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        if (direct) {
            std::reverse(temp.begin(), temp.end());
        }
        direct = ~direct;/*更换方向*/
        res.push_back(std::move(temp));
    }
    return res;
}
```

#### [107. 二叉树的层序遍历 II](https://leetcode-cn.com/problems/binary-tree-level-order-traversal-ii/)

```c_cpp
/*在102的基础上将结果reverse*/
vector<vector<int>> levelOrderBottom(TreeNode *root) {
    if (root == nullptr)
        return {};
    queue<TreeNode *> queue;
    queue.push(root);
    vector<vector<int>> res;
    while (!queue.empty()) {
        int size = queue.size();
        vector<int> temp(size);
        for (int i = 0; i < size; ++i) {
            auto p = queue.front();
            queue.pop();
            temp[i] = p->val;
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        res.push_back(std::move(temp));
    }
    std::reverse(res.begin(), res.end());
    return res;
}
```

#### [111. 二叉树的最小深度](https://leetcode-cn.com/problems/minimum-depth-of-binary-tree/)

```c_cpp
int minDepth(TreeNode *root) {
    //    return BFS(root);
    return DFS(root);
}
int BFS(TreeNode *root) {
    if (root == nullptr)
        return 0;
    queue<TreeNode *> queue;
    queue.push(root);
    int ret = 0;
    while (!queue.empty()) {
        ret++;
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            auto p = queue.front();
            queue.pop();
            /*遇到第一个叶子节点，其所在层次即最小高度*/
            if (p->left == nullptr && p->right == nullptr)
                return ret;
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
    }
    return ret;
}
int DFS(TreeNode *root) {
    if (root == nullptr)
        return 0;
    if (root->left == nullptr && root->right == nullptr)
        return 1;
    else if (root->left == nullptr)
        return DFS(root->right) + 1;
    else if (root->right == nullptr)
        return DFS(root->left) + 1;
    /*取最小层数*/
    return min(DFS(root->left), DFS(root->right)) + 1;
}
```

#### [126. 单词接龙 II](https://leetcode-cn.com/problems/word-ladder-ii/)

```c_cpp
/*思路：先广度遍历构建图，然后深度遍历回溯找到路径*/
vector<vector<string>> findLadders(string beginWord, string endWord, vector<string> &wordList) {
    vector<vector<string>> res;
    unordered_set<string> dict(wordList.begin(), wordList.end());
    if (dict.count(endWord) == 0) {/*没有结尾单词*/
        return res;
    }
    dict.erase(beginWord);/*删除头单词，避免重复*/
    unordered_map<string, set<string>> from;/*记录当前单词是有哪些单词转换过来的*/
    unordered_map<string, int> steps;/*记录当前单词属于那个曾，用于不免不必要的线路，同层次的转换不记录*/
    bool isFound = BFS(from, steps, dict, beginWord, endWord);

    if (isFound) {
        vector<string> path;
        path.push_back(endWord);
        DFS(beginWord, endWord, from, res, path);
    }
    return res;
}

bool BFS(unordered_map<string, set<string>> &from,
         unordered_map<string, int> &steps,
         unordered_set<string> &dict,
         const string &beginWord, const string &endWord) {
    int wordLen = beginWord.size();
    int step = 0;
    bool found = false;
    queue<string> queue;
    queue.push(beginWord);
    while (!queue.empty()) {
        step++;
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            auto currWord = std::move(queue.front());
            queue.pop();
            auto nextWord = currWord;
            for (int j = 0; j < wordLen; ++j) {
                char origin = nextWord[j];
                for (int c = 'a'; c <= 'z'; ++c) {
                    nextWord[j] = c;
                    if (steps.count(nextWord) && step == steps.at(nextWord)) {/*若转换的单词在之前出现过，且在当前单词的下一层才记录*/
                        from[nextWord].insert(currWord);
                    }
                    if (dict.find(nextWord) == dict.end())
                        continue;
                    dict.erase(nextWord);
                    queue.push(nextWord);
                    from[nextWord].insert(currWord);
                    steps[nextWord] = step;
                    if (nextWord == endWord)
                        found = true;
                }
                nextWord[j] = origin;
            }
        }
        if (found)
            break;
    }
    return found;
}
void DFS(const string &beginWord,
         string curWord,
         unordered_map<string, set<string>> &from,
         vector<vector<string>> &res,
         vector<string> &path) {
    /*从endWord开始王前找，结果需要reverse*/
    if (curWord == beginWord) {
        res.emplace_back(path.rbegin(), path.rend());
        return;
    }

    for (const auto &precursor : from[curWord]) {
        path.push_back(precursor);
        DFS(beginWord, precursor, from, res, path);
        path.pop_back();
    }
}
```

#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

```c_cpp
int ladderLength(string beginWord, string endWord, vector<string> &wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    /*若五endWord则无法转换*/
    if (dict.count(endWord) == 0) {
        return 0;
    }
    dict.erase(beginWord);
    /*图的最短路径，BFS*/
    unordered_set<string> visited;
    queue<string> queue;
    visited.insert(beginWord);
    queue.push(beginWord);
    int steps = 1;
    int wordLen = beginWord.size();
    while (!queue.empty()) {
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            string currWord = queue.front();
            queue.pop();
            /*对单词的每一位，从a-z遍历，找到可转换的单词*/
            string nextWord = currWord;
            for (int j = 0; j < wordLen; ++j) {
                char origin = nextWord[j];/*记录原字符,之后需要复原*/
                for (int c = 'a'; c <= 'z'; ++c) {
                    nextWord[j] = c;
                    if (c == origin)
                        continue;
                    if (dict.count(nextWord)) {
                        if (nextWord == endWord)
                            return steps + 1;/*可以转变到结尾单词，需要+1步*/
                        /*如果转换的单词没有遍历过，则入队，标记已访问*/
                        if (!visited.count(nextWord)) {
                            queue.push(nextWord);
                            visited.insert(nextWord);
                        }
                    }
                }
                nextWord[j] = origin;/*复原*/
            }
        }
        steps++;
    }
    return 0;
}
```

#### [130. 被围绕的区域](https://leetcode-cn.com/problems/surrounded-regions/)

```c_cpp
/*BFS*/
void solve(vector<vector<char>> &board) {
    int m = board.size(), n = board[0].size();
    queue<pair<int, int>> queue;
    /*左右边界,为O的入队*/
    for (int i = 0; i < m; ++i) {
        if (board[i][0] == 'O') {
            queue.emplace(i, 0);
            board[i][0] = 'A';
        }
        if (board[i][n - 1] == 'O') {
            queue.emplace(i, n - 1);
            board[i][n - 1] = 'A';
        }
    }
    /*上下边界为O的入队*/
    for (int i = 1; i < n - 1; ++i) {
        if (board[0][i] == 'O') {
            queue.emplace(0, i);
            board[0][i] = 'A';
        }
        if (board[m - 1][i] == 'O') {
            queue.emplace(m - 1, i);
            board[m - 1][i] = 'A';
        }
    }
    vector<vector<int>> direct = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    while (!queue.empty()) {
        auto[x, y]= queue.front();
        queue.pop();
        for (int i = 0; i < 4; ++i) {/*四个方向*/
            int nx = x + direct[i][0], ny = y + direct[i][1];
            if (nx < 0 || nx >= m || ny < 0 || ny >= n || board[nx][ny] != 'O')
                continue;
            /*属于同一个联通分量*/
            board[nx][ny] = 'A';
            queue.emplace(nx, ny);
        }
    }
    /*修改所有的O,并且将标记修改为原来的*/
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O')
                board[i][j] = 'X';
            if (board[i][j] == 'A')
                board[i][j] = 'O';
        }
    }
}
/*DFS,深度遍历将与边框连接的O进行标记*/
void solve1(vector<vector<char>> &board) {
    int m = board.size(), n = board[0].size();
    /*左右边界*/
    for (int i = 0; i < m; ++i) {
        DFS(board, i, 0);
        DFS(board, i, n - 1);
    }
    /*上下边界*/
    for (int i = 1; i < n - 1; ++i) {
        DFS(board, 0, i);
        DFS(board, m - 1, i);
    }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O')/*没标记的O就是被包围的*/
                board[i][j] = 'X';
            if (board[i][j] == 'A')/*将标记复原*/
                board[i][j] = 'O';
        }
    }
}
vector<vector<int>> direct = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
void DFS(vector<vector<char>> &board, int x, int y) {
    int m = board.size(), n = board[0].size();
    /*越界,不是O或以及搜索过*/
    if (x < 0 || x >= m || y < 0 || y >= n || board[x][y] != 'O' || board[x][y] == 'A')
        return;
    board[x][y] = 'A';
    /*四个方向出发,将相邻的O进行标记*/
    for (int i = 0; i < 4; ++i) {
        int nx = x + direct[i][0], ny = y + direct[i][1];
        DFS(board, nx, ny);
    }
}
```

#### [199. 二叉树的右视图](https://leetcode-cn.com/problems/binary-tree-right-side-view/)

```c_cpp
vector<int> rightSideView(TreeNode *root) {
    /*BFS*/
    vector<int> res;
    if (root == nullptr)
        return res;
    queue<TreeNode *> queue;
    queue.push(root);
    while (!queue.empty()) {
        int size = queue.size();
        TreeNode *p = nullptr;/*用于记录每层的node,遍历完后就是最后一个*/
        for (int i = 0; i < size; ++i) {
            p = queue.front();
            queue.pop();
            if (p->left)
                queue.push(p->left);
            if (p->right)
                queue.push(p->right);
        }
        res.push_back(p->val);
    }
    return res;
}
```

#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```c_cpp
/*BFS*/
int m = grid.size(), n = grid[0].size();
vector<vector<int>> direct{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
int res = 0;
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        if (grid[i][j] == '1') {
            res++;
            grid[i][j] = '0';
            queue<pair<int, int>> queue;
            queue.push({i, j});
            while (!queue.empty()) {
                auto[x, y] = queue.front();
                queue.pop();
                for (int k = 0; k < 4; ++k) {
                    int nx = x + direct[k][0], ny = y + direct[k][1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == '1') {
                        queue.push({nx, ny});
                        grid[nx][ny] = '0';
                    }
                }
            }
        }
    }
}
return res;
}
int numIslands1(vector<vector<char>> &grid) {
    int m = grid.size(), n = grid[0].size();
    int res = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (grid[i][j] == '1') {/*DFS将其相连的土地进行标记*/
                DFS(grid, i, j);
                res++;
            }
        }
    }
    return res;
}
vector<vector<int>> direct{{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
void DFS(vector<vector<char>> &grid, int x, int y) {
    if (grid[x][y] != '1')
        return;
    int m = grid.size(), n = grid[0].size();
    grid[x][y] = '0';
    for (int i = 0; i < 4; ++i) {
        int nx = x + direct[i][0], ny = y + direct[i][1];
        if (nx >= 0 && nx < m && ny >= 0 && ny < n)
            DFS(grid, nx, ny);
    }
}
```

#### [279. 完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```c_cpp
  int numSquares(int n) {
    vector<int> dp(n + 1);
    for (int i = 1; i <= n; ++i) {
      dp[i] = i;/*最坏情况需要i个数*/
      for (int j = 1; j * j <= i; ++j) {
        // i-j*j 表示 当前i减去一个完全平方数后所需的数量在加+1
        dp[i] = min(dp[i], dp[i - j * j] + 1);
      }
    }
    return dp[n];
  }
```

#### [301. 删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

```c_cpp
vector<string> res;
bool isValid(string &str) {
    int cnt = 0;
    for (const auto &item : str) {
        if (item == '(')
            cnt++;
        else if (item == ')') {
            if (--cnt < 0)
                return false;
        }
    }
    return cnt == 0;
}
void BackTrack(string s, int start, int lCount, int rCount, int lRemove, int rRemove) {
    if (lRemove == 0 && rRemove == 0) {
        if (isValid(s)) {
            res.push_back(s);
        }
        return;
    }
    int size = s.size();
    for (int i = start; i < size; ++i) {
        if (i > start && s[i] == s[i - 1]) {/*去重，例如(((())),对于该字符串，前面的四个删除任意一个都是相同的*/
            /*记录括号的使用数量*/
            if (s[i] == '(')
                lCount++;
            else if (s[i] == ')')
                rCount++;
            continue;
        }
        if (lRemove + rRemove > size - i)/*剩余字符串不够删除*/
            return;
        /*删除一个左括号*/
        if (lRemove > 0 && s[i] == '(')
            BackTrack(s.substr(0, i) + s.substr(i + 1), i, lCount, rCount, lRemove - 1, rRemove);
        /*删除一个右括号*/
        if (rRemove > 0 && s[i] == ')')
            BackTrack(s.substr(0, i) + s.substr(i + 1), i, lCount, rCount, lRemove, rRemove - 1);
        /*计算括号的使用数量*/
        if (s[i] == '(')
            lCount++;
        else if (s[i] == ')')
            rCount++;
        /*右括号大于左括号不合法*/
        if (rCount > lCount)
            break;
    }
}
vector<string> removeInvalidParentheses(string s) {
    int lRemove = 0, rRemove = 0;
    for (const auto &item : s) {
        if (item == '(')
            lRemove++;
        else if (item == ')') {
            if (lRemove == 0)
                rRemove++;
            else
                lRemove--;
        }
    }
    BackTrack(s, 0, 0, 0, lRemove, rRemove);
    return res;
}
```

#### [310. 最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

```c_cpp
/*图+BFS:反向BFS*/
vector<int> findMinHeightTrees(int n, vector<vector<int>> &edges) {
    if (n == 1)
        return {0};
    /*key为节点data为与自己相连的节点*/
    vector<vector<int>> adjacent(n);
    /*每个节点的度数*/
    vector<int> degree(n);
    /*构建图和记录节点度数*/
    for (const auto &edge : edges) {
        degree[edge[0]]++;
        degree[edge[1]]++;
        adjacent.at(edge[0]).push_back(edge[1]);
        adjacent.at(edge[1]).push_back(edge[0]);
    }
    /*将只有一个点相连的节点（即叶子节点）入队*/
    queue<int> queue;
    for (int i = 0; i < n; ++i) {
        if (degree[i] == 1)
            queue.push(i);
    }
    /*思路，从最外圈的叶子节点开始删除*/
    vector<int> res;
    while (!queue.empty()) {
        res.clear();/*清除之前一层的节点*/
        int size = queue.size();
        for (int i = 0; i < size; ++i) {
            int node = queue.front();
            queue.pop();
            res.push_back(node);/*记录当前出队的一整层*/
            degree[node]--;
            /*将与node相连的节点的度数都-1,若度数为1则入队*/
            for (const auto &i : adjacent.at(node)) {
                degree[i]--;
                if (degree[i] == 1)
                    queue.push(i);
            }
        }
    }
    return res;
}
```

#### [133. 克隆图](https://leetcode-cn.com/problems/clone-graph/)

```c_cpp
/*BFS*/
Node *cloneGraph(Node *node) {
    if (node == nullptr)
        return nullptr;
    unordered_map<Node *, Node *> visited;/*key为原始节点，data为克隆节点*/
    /*先克隆第一个节点*/
    visited[node] = new Node(node->val);
    queue<Node *> queue;
    queue.push(node);
    while (!queue.empty()) {
        auto needCol = queue.front();
        queue.pop();
        /*遍历所有邻居*/
        for (const auto &neighbor : needCol->neighbors) {
            /*判断邻居是否访问过，若没有则创建该邻居，并且入队*/
            if (visited.count(neighbor) == 0) {
                visited[neighbor] = new Node(neighbor->val);
                queue.push(neighbor);
            }
            /*更新当前克隆节点的neighbors*/
            visited[needCol]->neighbors.push_back(visited[neighbor]);
        }
    }
    return visited[node];
}
/*DFS*/
Node *cloneGraph1(Node *node) {
    return DFS(node);
}
unordered_map<Node *, Node *> visited;/*key为源节点，data为克隆节点*/
Node *DFS(Node *node) {
    if (node == nullptr)
        return nullptr;

    if (visited.count(node))/*如果以及访问过则直接返回*/
        return visited[node];
    /*克隆节点*/
    auto colNode = new Node(node->val);
    /*递归的复制neighbor节点*/
    for (const auto &n : node->neighbors) {
        colNode->neighbors.push_back(DFS(n));
    }
    return colNode;
}
```

## 三 、DP

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```c_cpp
string longestPalindrome(string s) {
    if (s.size() == 1)
        return s;
    if (s.size() == 2 && s[0] == s[1])
        return s;
    /*dp[i][j]：表示i...j是否为回文串*/
    int size = s.size();
    vector<vector<bool>> dp(size, vector<bool>(size));
    /*每一个字符本身就是回文串*/
    for (int i = 0; i < size; ++i) {
        dp[i][i] = true;
    }
    int len = 1;
    int start = 0;
    for (int right = 1; right < size; ++right) {
        for (int left = 0; left < right; ++left) {
            if (s[left] == s[right]) {
                if (right - left + 1 <= 3)/*两边相等且长度<=则为回文串*/
                    dp[left][right] = true;
                else/*若>3,则看更进一步是否为回文串*/
                    dp[left][right] = dp[left + 1][right - 1];

                if (dp[left][right]) {/*若是回文串，则比较更新答案的长度和起始位置*/
                    if (right - left + 1 > len) {
                        len = right - left + 1;
                        start = left;
                    }
                }
            }
        }
    }
    return s.substr(start, len);
}
```

#### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

```c_cpp
bool isMatch(string s, string p) {
    int m = s.size(), n = p.size();
    /*Dp[i][j] 代表 s[i]和p[j]是匹配*/
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));/*多一行一列用于空字符的匹配*/
    dp[0][0] = true;/*空字符与空字符匹配*/
    /*第一行第一列为边界，第一列与空字符匹配都为false(vector默认false),第一列需要初始化*/
    for (int j = 2; j <= n; j += 2) {/*若p的偶数个为*则可以尝试与空字符匹配*/
        dp[0][j] = (p[j - 1] == '*' && dp[0][j - 2]);
    }
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (p[j - 1] == '*') {
                /*情况一，*重复0次，则看前两个字符位置是否匹配*/
                /*情况二，*重复前一个字符.则看前一个字符是否匹配，以及之前的和s[i-1]和p[j]是否匹配，若都匹配则可以复制前一个字符使用*/
                dp[i][j] = dp[i][j - 2] || ((s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
            } else {
                /*当前字符不为*则看前一个字符是否匹配和当前字符是否匹配*/
                dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
            }
        }
    }
    return dp[m][n];
}
/*递归*/
bool isMatch1(string s, string p) {
    int len1 = s.size();
    int len2 = p.size();
    /*出口*/
    if (len2 == 0)
        return len1 == 0;
    /*判断第一个字符是否相等，若p[0]==‘。’ 直接相等*/
    bool firstMatch = (len1 > 0 && len2 > 0 && (s[0] == p[0] || p[0] == '.'));
    /*‘*’右两种情况*/
    if (len2 > 1 && p[1] == '*') {
        /*情况一：*号不产生作用，使得前一个字符重复0次*/
        /*情况二：*号产生作用，使得前一个字符重复,前提前一个字符需匹配*/
        return isMatch(s, p.substr(2)) || (firstMatch && isMatch(s.substr(1), p));
    }
    /*继续判断后面的是否匹配*/
    return firstMatch && isMatch(s.substr(1), p.substr(1));
}
```

#### [44. 通配符匹配](https://leetcode-cn.com/problems/wildcard-matching/)

```c_cpp
bool isMatch(string s, string p) {
  int m = s.size(), n = p.size();
  /*定义：dp[i...j]是否能匹配*/
  vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
  dp[0][0] = true;/*空字符与空字符匹配*/
  /*第一列与空字符不匹配默认false，第一行与空字符进进行匹配，若有*才能匹配(初始化边界条件)*/
  for (int j = 1; j <= n; ++j) {
    if (p[j - 1] == '*')
      dp[0][j] = true;
    else
      break;
  };
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (p[j - 1] == '*') {
        /*情况一：当*为空，不使用，则看dp[i][j-1]是否能匹配*/
        /*情况二：当*不为空，使用进行多次匹配，则看dp[i-1][j]是否能匹配*/
        dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
      } else if (s[i - 1] == p[j - 1] || p[j - 1] == '?')/*当前字符相等或p[j-1]为单个万能符，则只看前一次是否能匹配*/
        dp[i][j] = dp[i - 1][j - 1];
    }
  }
  return dp[m][n];
}
```

#### [72. 编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```c_cpp
int minDistance(string word1, string word2) {
  int m = word1.size(), n = word2.size();
  /*dp[i...j]从 i字符到j字符需要变换的最小次数*/
  vector<vector<int>> dp(m + 1, vector<int>(n + 1));
  /*初始化边界，空字符到第一列需要删除i次，空字符到第一行需要新增j次*/
  for (int i = 0; i <= m; ++i) {
    dp[i][0] = i;
  }
  for (int j = 0; j <= n; ++j) {
    dp[0][j] = j;
  }
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (word1[i - 1] == word2[j - 1]) {/*字符相同，则直接从上一步过来不用加*/
        dp[i][j] = dp[i - 1][j - 1];
      } else {
        dp[i][j] = min(min(dp[i - 1][j - 1], dp[i - 1][j]), dp[i][j - 1]) + 1;/*从上一步的改，删，加挑选最小操作然后+1*/
      }
    }
  }
  return dp[m][n];
}
```

 #### [97.交错字符串](https://leetcode-cn.com/problems/interleaving-string/)

```c_cpp
bool isInterleave(string s1, string s2, string s3) {
  int m = s1.size(), n = s2.size();
  //dp[i..j] 表示 s1的前i位+s2前j位是否能组成 s3的前i+j位
  vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));
  if (m + n != s3.size())
    return false;
  dp[0][0] = true;//边界条件
  for (int i = 1; i <= m; ++i) {//初始化第一列,表示当前s1当前字符i能否构成s3前i位
    dp[i][0] = dp[i - 1][0] && s1[i - 1] == s3[i - 1];
  }
  for (int j = 1; j <= n; ++j) {//初始化第一行，表示当前s2当前字符j能否构成s3前j位
    dp[0][j] = dp[0][j - 1] && s2[j - 1] == s3[j - 1];
  }
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      int p = i + j - 1;
      /*第一种情况，当前s3[i+j] == s1[i],则看前面的 i-1 + j 是否能组合,即 dp[i-1][j] 是否为true
      /*第二种情况，当前s3[i+j] == s2[j],则看前面的 i + j-1 是否能组合,即 dp[i-1][j] 是否为true
       * */
      dp[i][j] = (s1[i - 1] == s3[p] && dp[i - 1][j]) || (s2[j - 1] == s3[p] && dp[i][j - 1]);
    }
  }
  return dp[m][n];
}
```

#### [115.不同的子序列](https://leetcode-cn.com/problems/distinct-subsequences/)  画图作

```c_cpp
int numDistinct(string s, string t) {
  int m = s.size(), n = t.size();
  if (m < n)//无法构成子串匹配
    return 0;
  //dp定义： dp[i][j]表示 s[i:]子序中出现t[j:]的个数
  vector<vector<unsigned long long>> dp(m + 1, vector<unsigned long long>(n + 1));
  //边界条件
  //当j==n时，t[j:] 为空字符串是所有字符串的子串，因此对于0<=i<=m,有 dp[i][n]=1;
  //当i==m，且j<n时，s[i:] 为空字符串,t[j:]非空子串，无法匹配，因此对于 i==m,且j<n是 dp[m][j]=0;
  for (int i = 0; i <= m; ++i) {
    dp[i][n] = 1;
  }

  for (int i = m - 1; i >= 0; --i) {
    for (int j = n - 1; j >= 0; --j) {
      if (s[i] == t[j]) {
        //两种情况，第一个中是匹配，记录s[i+1:}与t[j+1:] 匹配的此时，若不相同则，记录s[i+1:]（相当于删掉s当前的字符）与t[j:]匹配的次数
        dp[i][j] = dp[i + 1][j + 1] + dp[i + 1][j];
      } else {
        //不匹配直接删除s当前字符
        dp[i][j] = dp[i + 1][j];
      }
    }
  }
  return dp[0][0];
}
```

#### [53.最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

```c_cpp
int maxSubArray(vector<int> &nums) {
  int n = nums.size();
  /*定义，dp[i]为以i结尾的当前子序的最大值,由于只需前一个的值，因此可以优化dp数组*/
  int max_num = nums[0];
  int p = nums[0];

  for (int i = 1; i < n; ++i) {
    int q = max(p, 0) + nums[i];//若dp[i-1]的值小于0直接舍弃，从当前重新开始
    if (q > max_num)
      max_num = q;
    //更新值
    p = q;
  }
  return max_num;
}
```

#### [62.不同路径](https://leetcode-cn.com/problems/unique-paths/)

```c_cpp
int uniquePaths(int m, int n) {
  //dp[i][j]表示抵达当前位置的路径数量
  vector<vector<int>> dp(m, vector<int>(n));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == 0 || j == 0)//边界条件，当i==0时只能从上方下来，j==0时只能从左边过来，因此只有一条路径
        dp[i][j] = 1;
      else
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1];//当前位置可以从上或左边走过来
    }
  }
  return dp[m - 1][n - 1];
}
```

#### [63.不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

```c_cpp
int uniquePathsWithObstacles(vector<vector<int>> &obstacleGrid) {
  int m = obstacleGrid.size(), n = obstacleGrid[0].size();
  if (obstacleGrid[m - 1][n - 1] == 1)
    return 0;
  /*定义dp[i][j]表示抵达当前位置的路径数量*/
  vector<vector<int>> dp(m, vector<int>(n));
  //边界条件,当i==0或j==0时，只能从上或左走，只有一条路径，若出现障碍后则都不可以走,为0；
  for (int i = 0; i < m && obstacleGrid[i][0] != 1; ++i) {
    dp[i][0] = 1;
  }
  for (int j = 0; j < n && obstacleGrid[0][j] != 1; ++j) {
    dp[0][j] = 1;
  }
  for (int i = 1; i < m; ++i) {
    for (int j = 1; j < n; ++j) {
      dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j] + dp[i][j - 1];//当前位置障碍则不计算，若无障碍则可以从上和左抵达当前位置
    }
  }
  return dp[m - 1][n - 1];
}
```

#### [64.最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

```c_cpp
int minPathSum(vector<vector<int>> &grid) {
  int m = grid.size(), n = grid[0].size();
  //dp[i][j]为当前路径最小值
  vector<vector<int>> dp(m, vector<int>(n));
  //边界条件,第一行和第一列只有一个方向，因此只要叠加前面的路径值
  dp[0][0] = grid[0][0];
  for (int i = 1; i < m; ++i) {
    dp[i][0] = grid[i][0] + dp[i - 1][0];
  }
  for (int j = 1; j < n; ++j) {
    dp[0][j] = grid[0][j] + dp[0][j - 1];
  }
  for (int i = 1; i < m; ++i) {
    for (int j = 1; j < n; ++j) {
      dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];//左上取最小路径+当前路径值
    }
  }
  return dp[m - 1][n - 1];
}
```

#### [70.爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```c_cpp
int climbStairs(int n) {
//    if (n <= 2)
//      return n;
//    //dp[i]为抵达i阶台阶的方法数量
//    vector<int> dp(n + 1);
//    //初始换边界条件
//    dp[1] = 1;
//    dp[2] = 2;
//    for (int i = 3; i <= n; ++i) {
//      dp[i] = dp[i - 2] + dp[i - 1];//抵达当前i阶梯可以从i-1和i-2阶梯
//    }
//    return dp[n];

  //优化版本，由于只需要前两个值可以使用两个变量代替dp数组
  if (n <= 2)
    return n;
  int n1 = 1, n2 = 2;
  int ans = 0;
  for (int i = 3; i <= n; ++i) {
    ans = n1 + n2;
    n1 = n2;
    n2 = ans;
  }
  return ans;
}
```

#### [91.解码方法](https://leetcode-cn.com/problems/decode-ways/)

```c_cpp
int numDecodings1(string s) {
  /*跳台阶升级版，满足条件就可以调一个字符或两个字符，细节注意：dp[0]=1为空字符串有一种解，注意遍历的字符位置*/
  int n = s.size();
  //dp[0...i]为0...i的编码数量
  vector<int> dp(n + 1);
  dp[0] = 1;//经、空字符串有一个解  注意
  for (int i = 1; i <= n; ++i) {
    if (s[i - 1] != '0') {//当前字符不为0时，可以单独使用进行编码
      dp[i] += dp[i - 1];
    }
    if (i > 1 && s[i - 2] != '0' && (s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26) {//使用两个字符编码,且值<=26可以两个进行编码
      dp[i] += dp[i - 2];
    }
  }
  return dp[n];
}
//优化版本
int numDecodings(string s) {
  /*跳台阶升级版，满足条件就可以调一个字符或两个字符，细节注意：dp[0]=1为空字符串有一种解，注意遍历的字符位置*/
  int n = s.size();
  //a =dp[i-1] b=dp[i-1] c=dp[i]
  int a = 0, b = 1, c;
  for (int i = 1; i <= n; ++i) {
    c = 0;
    if (s[i - 1] != '0') {//当前字符不为0时，可以单独使用进行编码
      c += b;
    }
    if (i > 1 && s[i - 2] != '0' && (s[i - 2] - '0') * 10 + (s[i - 1] - '0') <= 26) {//使用两个字符编码,且值<=26可以两个进行编码
      c += a;
    }
    tie(a, b) = {b, c};
  }
  return c;
}
```

#### [96.不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```c_cpp
int numTrees(int n) {
//    大问题中找到子问题：求G(n)，即求[1,n]的解的和，那就先以其中的i(1<=i<=n)为顶点, 记为f(i)
//    解决子问题：i为顶点的解，为G[i-1] * G[n-i]的解。
//    合并子问题求的大问题的解: G[n] = f(1) +....+f(i) + ...f(n)
  vector<int> dp(n + 1);//G(n)
  dp[0] = 1;//空树
  dp[1] = 1;//一个节点只有一种树；
  for (int i = 2; i <= n; ++i) {
    for (int j = 1; j <= i; ++j) { //f(i)
      dp[i] += dp[j - 1] * dp[i - j];  
    }
  }
  return dp[n];
}
```

#### [120.三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

```c_cpp
int minimumTotal1(vector<vector<int>> &triangle) {
  //自底向上遍历
  vector<vector<int>> dp(triangle);
  for (int i = dp.size() - 2; i >= 0; --i) {
    for (int j = 0; j <= i; ++j) {
      //当前位置+ 相邻位置取最小值
      dp[i][j] = triangle[i][j] + min(dp[i + 1][j], dp[i + 1][j + 1]);
    }
  }
  return dp[0][0];
}
//优化版 内存
int minimumTotal(vector<vector<int>> &triangle) {
  //自底向上遍历
  vector<int> dp(triangle.back());
  for (int i = triangle.size() - 2; i >= 0; --i) {//倒数第二层开始
    for (int j = 0; j <= i; ++j) {
      //当前位置+ 相邻位置取最小值
      dp[j] = triangle[i][j] + min(dp[j], dp[j + 1]);//覆盖dp数组
    }
  }
  return dp[0];
}
```

#### [152.乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

```c_cpp
int maxProduct1(vector<int> &nums) {
  //考虑当前为负数的话，就期望之前结尾的也为负数，因此增加一个数组记录最小乘积
  int n = nums.size();
  vector<int> MaxNum(n), MinNum(n);
  MaxNum[0] = nums[0];
  MinNum[0] = nums[0];
  int ans = nums[0];
  for (int i = 1; i < n; ++i) {
    //取最大值= 前一个值*当前值，若当前值为负，则需考虑*最小值,在于当前值进度对比
    MaxNum[i] = max(MaxNum[i - 1] * nums[i], max(MinNum[i - 1] * nums[i], nums[i]));
    //取最小值
    MinNum[i] = min(MinNum[i - 1] * nums[i], min(MaxNum[i - 1] * nums[i], nums[i]));
    ans = max(ans, MaxNum[i]);
  }
  return ans;
}
//优化版
int maxProduct(vector<int> &nums) {
  //考虑当前为负数的话，就期望之前结尾的也为负数，因此增加一个数组记录最小乘积
  int n = nums.size();
  int MaxNum = nums[0], MinNum = nums[0], ans = nums[0];
  for (int i = 1; i < n; ++i) {
    int mx = MaxNum, mn = MinNum;
    //取最大值= 前一个值*当前值，若当前值为负，则需考虑*最小值,在于当前值进度对比
    MaxNum = max(mx * nums[i], max(mn * nums[i], nums[i]));
    //取最小值
    MinNum = min(mn * nums[i], min(mx * nums[i], nums[i]));
    ans = max(ans, MaxNum);
  }
  return ans;
}
```

#### [300.最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

```c_cpp
int lengthOfLIS(vector<int> &nums) {
  int n = nums.size();
  //dp[i]表示以 i结尾位置的最长子序长度
  int ans = 1;
  vector<int> dp(n, 1);//每个数字本身就是以自己结尾的长度为1；
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      if (nums[j] < nums[i]) {//若有比前面有比自己小的字符则取最大值 dp[j]+1 包含自己
        dp[i] = max(dp[j] + 1, dp[i]);
      }
    }
    ans = max(dp[i], ans);
  }
  return ans;
}
```

#### [312.戳气球](https://leetcode-cn.com/problems/burst-balloons/)

```c_cpp
int maxCoins(vector<int> &nums) {
  int n = nums.size();
  //修改原始数组，在头尾插入1方便计算
  nums.insert(nums.begin(), 1);
  nums.push_back(1);
  //dp[i][j] 表示(i..j)不包括i和j可以获得的最大硬币数量
  vector<vector<int>> dp(n + 2, vector<int>(n + 2));
  //不断的扩大边界
  for (int i = n - 1; i >= 0; --i) {//左边界
    for (int j = i + 1; j <= n + 1; ++j) {//有边界
      for (int k = i + 1; k < j; ++k) {//k在(i,j)中间，尝试戳破k，取其中的最大值
        int currCoins = nums[i] * nums[k] * nums[j];//错破最后一个气球k获得的硬币数量
        int currTotalCoins = dp[i][k] + currCoins + dp[k][j];//当k为最后一个气球，戳爆后可以获得总硬币数量
        dp[i][j] = max(dp[i][j], currTotalCoins);
      }
    }
  }
  return dp[0][n + 1];
}
```

[322.零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```c_cpp
int coinChange(vector<int> &coins, int amount) {
  //dp[i]表示为i最少需要的硬币数量
  int Max = amount + 1;
  vector<int> dp(amount + 1, Max);
  //边界条件0元只有0种兑换方法，其他标记位Max(代表无法凑出的情况)
  dp[0] = 0;
  std::sort(coins.begin(), coins.end());
  for (int i = 1; i <= amount; ++i) {
    //每次选择一个硬币coin, i-coin 代表减去硬币后所需的硬币数量，然后在加当前硬币+1
    for (const auto &coin : coins) {
      if (coin <=i)//硬币要小于等于i才能尝试兑换
        dp[i] = min(dp[i], dp[i - coin] + 1);
    }
  }
  return dp[amount] == Max ? -1 : dp[amount];
}
```

#### [121.买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  //dp[i][0] 代表在第i天买入时的最大利润，dp[i][1]代表在第i天卖出时的最大利润
  vector<vector<int>> dp(n, vector<int>(2));
  //边界条件，第一天买入以及第一天卖出为0
  dp[0][0] = -prices[0];
  dp[0][1] = 0;
  int maxProfit = 0;
  for (int i = 1; i < n; ++i) {
    dp[i][0] = max(dp[i - 1][0], -prices[i]);//买入时的最大利润为，max(昨天买入，今天买入)
    dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);//卖出时的最大利润为，max(昨天卖出，昨天持有+今天股价)
  }
  return dp[n - 1][1];
}
```

#### [122.买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  //dp[i][0] 代表在i天持有的最大利润，dp[i][1]代表在第i天时不持有的最大利润
  vector<vector<int>> dp(n, vector<int>(2));
  dp[0][0] = -prices[0];
  dp[0][1] = 0;
  for (int i = 1; i < n; ++i) {
    //第i天买持有最大利润=max(昨天已经持有股票，昨天未持有股票，今天买入)
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] - prices[i]);
    //第i天不持有的最大利润=max(昨天已经不持有，昨天持有，今天卖出)
    dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);
  }
  return dp[n - 1][1];
}
```

#### [123.买卖股票的最佳时机 III](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iii/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  //dp[i][k][0]代表不持有股票的最大利益，dp[i][k][1]代表买入股票的利益；
  /*
   * 第一个：天数，
   * 第二个:交易次数，题目要求两次，因此需要取到3
   * 第三个:为状态，0 empty 1 full
   * */
  vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(3, vector<int>(2)));

  for (int k = 0; k < 3; ++k) {
    dp[0][k][0] = 0;//第0天不持有股票，不管交易几次，都没有利润
    dp[0][k][1] = INT_MIN;//第0天没有开始，不管交易几次都无法买入，使用INT_MIN标记
  }

  int maxAns;
  for (int i = 1; i <= n; ++i) {
    for (int k = 1; k < 3; ++k) {//第一次交易开始
      //今天不持有股票的最大利润为 max(昨天已经不持有股票，昨天持有股票,今天卖出)
      dp[i][k][0] = max(dp[i - 1][k][0], dp[i - 1][k][1] + prices[i - 1]);
      //今天持有股票的最大利润为 max(昨天已经持有股票，昨天每持有股票,今天买入)
      //买入时计算交易次数，因此当昨天进行到k-1次没有持有股票,今天才可以买入
      dp[i][k][1] = max(dp[i - 1][k][1], dp[i - 1][k - 1][0] - prices[i - 1]);

      maxAns = max(maxAns, dp[i][k][0]);
    }
  }
  return maxAns;
}
```

#### [188.买卖股票的最佳时机 IV](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-iv/)

```c_cpp
int maxProfit(int k, vector<int> &prices) {
  int n = prices.size();
  if (k > n / 2)//两天完成一笔交易
    k = n / 2;
  /*
   * n:天数
   * k:交易次数
   * status: 0 empty 1 full
   * */
  vector<vector<vector<int>>> dp(n + 1, vector<vector<int>>(k + 1, vector<int>(2)));
  for (int nk = 0; nk < k + 1; ++nk) {
    dp[0][nk][0] = 0;//第0天，不管交易几次，都是0
    dp[0][nk][1] = INT_MIN;//第0天还有买入，不可能卖出，使用INT_MIN标记
  }

  int maxAns = 0;
  for (int i = 1; i <= n; ++i) {
    for (int nk = 1; nk <= k; ++nk) {
      //今天不持有最大利润= max(昨天已经不持有， 昨天持有，今天卖出)
      dp[i][nk][0] = max(dp[i - 1][nk][0], dp[i - 1][nk][1] + prices[i - 1]);
      //今天持有的最大丽人=max(昨天以及持有， 昨天的nk-1次交易不持有，今天买入)
      //买入的时候计算交易次数，因此需要昨天进行到nk-1次交易，今天才有交易机会
      dp[i][nk][1] = max(dp[i - 1][nk][1], dp[i - 1][nk - 1][0] - prices[i - 1]);

      maxAns = max(dp[i][nk][0], maxAns);
    }
  }
  return maxAns;
}
```

#### [309.最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  /*
   * dp[i][0] 手中持有股票的最大利润
   * dp[i][1] 手中不持有股票,且处于冷冻期的最大利润
   * dp[i][2] 手册不持有股票，且不处于冷冻期的最大利润
   * */
  vector<vector<int>> dp(n, vector<int>(3));
  dp[0][0] = -prices[0];

  for (int i = 1; i < n; ++i) {
    //第i天持有股票最大利润 = max(昨天就持有，昨天不持有不在冷冻期且今天买入）
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
    //第i天不持有股票且在冷冻期（今天要卖掉）, 昨天持有股票+今天卖出价格
    dp[i][1] = dp[i - 1][0] + prices[i];
    //第i天不持有股票，且不在冷冻期(今天不进行任何操作) = max(昨天已经不持有不在冷冻，昨天卖出今天冷冻)
    dp[i][2] = max(dp[i - 1][2], dp[i - 1][1]);
  }
  return max(dp[n - 1][1], dp[n - 1][2]);
}
```

#### [198.打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```c_cpp
int rob(vector<int> &nums) {
  int n = nums.size();
  if (n == 1)//特殊情况处理
    return nums[0];
  //dp[i]表示i当前可以抢的最大金额
  vector<int> dp(n);
  //边界条件
  dp[0] = nums[0];//只有一间房子只能抢它
  dp[1] = max(nums[0], nums[1]);//两个相连的房子，只能抢一个，抢最大的那个
  for (int i = 2; i < n; ++i) {
    //抢不抢当前的房子有前面两个房子决定，max(抢不相连的房子和当前房子，抢前一个房子)
    dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
  }
  return dp[n - 1];
}
```

#### [213.打家劫舍 II](https://leetcode-cn.com/problems/house-robber-ii/)

```c_cpp
int rob(vector<int> &nums) {
  //由于首尾相连，因此，若抢最后一个，则不能抢第一个，反之亦然。
  //因此有两个区间可以抢  nums[0,size-2] 和 nums[1,size-1]
  int n = nums.size();
  if (n < 2)
    return nums[0];
  int robFirst = help(nums, 0, n - 2);
  int robLast = help(nums, 1, n - 1);
  return max(robFirst, robLast);
}
int help(const vector<int> &num, int start, int end) {
  if (end - start + 1 == 1)//只有一间房子
    return num[start];
  //由于只需前两个房子的值，可以使用两个变量代表dp数组
  int a = num[start];// dp[i-2]
  int b = max(a, num[start + 1]);//dp[i-1]
  for (int i = start + 2; i <= end; ++i) {
    int c = max(num[i] + a, b);//抢不抢当前房子 =  max(抢当前房子+不相连的房子，相连的房子)
    tie(a, b) = {b, c};//更新值
  }
  return b;
}
```

#### [337.打家劫舍 III](https://leetcode-cn.com/problems/house-robber-iii/)

```c_cpp
struct subTree {
  int rob;
  int not_rob;
};
int rob(TreeNode *root) {
  auto sub = DFS(root);
  return max(sub.rob, sub.not_rob);
}
subTree DFS(TreeNode *root) {
  if (!root)
    return {0, 0};
  subTree left = DFS(root->left);
  subTree right = DFS(root->right);
  //抢当前节点，则无法抢子节点
  int rob = root->val + left.not_rob + right.not_rob;
  //不抢当前节点，则可以抢子节点，也可以不抢子节点， 取最大值
  int not_rob = max(left.rob, left.not_rob) + max(right.rob, right.not_rob);
  return {rob, not_rob};
}
```

#### [221.最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```c_cpp
int maximalSquare(vector<vector<char>> &matrix) {
  int m = matrix.size(), n = matrix[0].size();
  if (m == 0 || n == 0)
    return 0;
  //定义dp[i][j] 表示以 坐标(i,j)为右下角的最大边长
  vector<vector<int>> dp(m, vector<int>(n));
  int maxEdge = 0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (matrix[i][j] != '1')//当位置不能作为右下角
        continue;

      if (i == 0 || j == 0)//边界条件，第一行,第一列最大边长只能为1
        dp[i][j] = 1;
      else // 三个方向上取最小的边然后+1
        dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;

      maxEdge = max(maxEdge, dp[i][j]);
    }
  }
  return maxEdge * maxEdge;
}
```

#### [279.完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```c_cpp
int numSquares(int n)
  // 定义： i需要最小的平方数的数量
  vector<int> dp(n + 1);
  for (int i = 1; i <= n; ++i) {
    dp[i] = i;//最坏情况，需要i个1
    for (int j = 1; j * j <= i; ++j) {
      dp[i] = min(dp[i], dp[i - j * j] + 1);//i-j*j 表示减去一个完全平方数后所需要要的数量后再+1
    }
  }
  return dp[n];
}
```

#### [303.区域和检索 - 数组不可变](https://leetcode-cn.com/problems/range-sum-query-immutable/)

```c_cpp
class NumArray {
 public:
  NumArray(vector<int> &nums) {
    m_sumRanges.resize(nums.size());
    m_sumRanges[0] = nums[0];
    //每一位存储的是从0到自身的和
    for (int i = 1; i < nums.size(); ++i) {
      m_sumRanges[i] = nums[i] + m_sumRanges[i - 1];
    }
  }

  int sumRange(int left, int right) {
    return left == 0 ? m_sumRanges[right] : m_sumRanges[right] - m_sumRanges[left - 1];
  }
 private:
  vector<int> m_sumRanges;
};
```

#### [304.区域和检索 - 矩阵不可变](https://leetcode-cn.com/problems/range-sum-query-2d-immutable/)

```c_cpp
class NumMatrix {
  //一维前缀和
 public:
  NumMatrix(vector<vector<int>> &matrix) {
    int m = matrix.size();
    if (m > 0) {
      int n = matrix[0].size();
      if (n > 0) {
        sums = vector<vector<int>>(m, vector<int>(n + 1));//预留多一列用于处理0列的情况
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            sums[i][j + 1] = sums[i][j] + matrix[i][j];
          }
        }
      }
    }
  }

  int sumRegion(int row1, int col1, int row2, int col2) {
    int ans = 0;
    for (int i = row1; i <= row2; ++i) {
      ans += sums[i][col2 + 1] - sums[i][col1];//注意：sums数组多了一列，注意偏移量
    }
    return ans;
  }
 private:
  vector<vector<int>> sums;
};

class NumMatrix {
  //二维前缀和
 public:
  NumMatrix(vector<vector<int>> &matrix) {
    int m = matrix.size();
    if (m > 0) {
      int n = matrix[0].size();
      if (n > 0) {
        sums = vector<vector<int>>(m + 1, vector<int>(n + 1));
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            //减去重复的部分+当前位置
            sums[i + 1][j + 1] = sums[i][j + 1] + sums[i + 1][j] - sums[i][j] + matrix[i][j];
          }
        }
      }
    }
  }

  int sumRegion(int row1, int col1, int row2, int col2) {
    //田字形，已知整体面积，上面面积，左边面积，左上面积，求右下角矩形的面积。 右下角矩形的面积=整体面积-上面面积-左边面积+左上面积
    return sums[row2 + 1][col2 + 1] - sums[row1][col2 + 1] - sums[row2 + 1][col1] + sums[row1][col1];
  }
 private:
  vector<vector<int>> sums;
```

<br/>

<br/>

# 四、HOT 100

#### [1.两数之和](https://leetcode-cn.com/problems/two-sum/)

```c_cpp
  vector<int> twoSum(vector<int> &nums, int target) {
    unordered_map<int, int> map;//记录目标以及其坐标
    int n = nums.size();
    for (int i = 0; i < n; ++i) {
      int diff = target - nums[i];//计算组成目标数所需的值
      if (map.count(diff)) {//判断所需的值是否在表中，若在直接返回
        return {i, map[diff]};
      } else {//将出现过的数都放入表中
        map[nums[i]] = i;
      }
    }
    return {};
  }
```

#### [2.两数相加]((https://leetcode-cn.com/problems/add-two-numbers/))

```c_cpp
ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
  //dummy节点
  auto dummy = new ListNode(0);
  ListNode *p = dummy;
  int carry = 0;
  while (l1 || l2) {
    int num1 = l1 ? l1->val : 0;
    int num2 = l2 ? l2->val : 0;
    //计算和（包括进位）
    int sum = num1 + num2 + carry;
    //是计算进位
    carry = sum / 10;
    p->next = new ListNode(sum % 10);
    p = p->next;
    if (l1) {
      l1 = l1->next;
    }
    if (l2) {
      l2 = l2->next;
    }
  }
  //判断最后是否还有进位
  if (carry)
    p->next = new ListNode(carry);
  return dummy->next;

```

#### [3.无重复字符的最长子串](https://leetcode-cn.com/problems/longest-substring-without-repeating-characters/)

```c_cpp
int lengthOfLongestSubstring(string s) {
  //滑动窗口
  int n = s.size();
  int right = 0;
  int ans = 0;
  unordered_set<char> set;
  for (int left = 0; left < n; ++left) {
    while (right < n && set.count(s[right]) == 0) {//若没有重复则插入表中，直到遇到重复
      set.insert(s[right++]);
    }
    ans = max(ans, right - left);//遇到重复后，更新最大长度
    set.erase(s[left]);//删除最右侧的字符
  }
  return ans;
}
```

#### [4.寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

```c_cpp
double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
//二分查找
int n = nums1.size(), m = nums2.size();
if (m < n) {//确保第一个数组为较小那个,二分查找已较小的那个为基础
  return findMedianSortedArrays(nums2, nums1);
}
int size = n + m;
int left = 0, right = n;
int LMax1 = 0, LMax2 = 2, RMin1 = 0, RMin2 = 0;
while (left <= right) {
  int cut1 = (right - left + 1) / 2 + left;//第一个数组分割的位置
  //第二个数组分割的位置; 因为是找中位数所以，中位数的位置为size/2  由于数组1已经有cut1个数，因此数组2的切割位置 size/2-cut1
  int cut2 = size / 2 - cut1;
  LMax1 = cut1 == 0 ? INT_MIN : nums1[cut1 - 1];//第一个数组若切割到0的位置，则左侧没有数据，使用INT_MIN表示，确保足够小
  RMin1 = cut1 == n ? INT_MAX : nums1[cut1];//第一个数组若切割到n的位置，则右侧没有数据，使用INT_MAX表示，确保足够大
  LMax2 = cut2 == 0 ? INT_MIN : nums2[cut2 - 1];
  RMin2 = cut2 == m ? INT_MAX : nums2[cut2];
  if (LMax1 > RMin2) { /* 说明C2位置过小，需要扩大，C2由C1决定，而C1的大小取决于【lo,hi】,因此缩小hi*/
    right = cut1 - 1;
  } else if (LMax2 > RMin1) {/* 说明C1位置过小，需要扩大C1,而C1的大小取决于【lo,hi】,因此扩大lo*/
    left = cut1 + 1;
  } else/*若不满足上述情况，则找到了两个合适的分割点，退出即可*/
    break;
}
if (size % 2 != 0) {//奇数取中位数
  return min(RMin1, RMin2);
}
//偶数取中位数
return (max(LMax1, LMax2) + min(RMin1, RMin2)) / 2.0;
}

double findMedianSortedArrays1(vector<int> &nums1, vector<int> &nums2) {
  //合并两个数组，然后找到中位数
  int m = nums1.size(), n = nums2.size();
  //特殊情况判断
  if (m == 0) {
    return n % 2 == 0 ? (double) (nums2[n / 2] + nums2[n / 2 - 1]) / 2 : nums2[n / 2];
  }
  if (n == 0) {
    return m % 2 == 0 ? (double) (nums1[m / 2] + nums1[m / 2 - 1]) / 2 : nums1[m / 2];
  }
  //合并数组
  int size = m + n;
  vector<int> nums(size);
  int i = 0, j = 0, k = 0;
  while (k < size) {
    if (nums1[i] < nums2[j]) {
      nums[k++] = nums1[i++];
    } else {
      nums[k++] = nums2[j++];
    }
    if (i == m) {
      while (j < n) {
        nums[k++] = nums2[j++];
      }
    }
    if (j == n) {
      while (i < m) {
        nums[k++] = nums1[i++];
      }
    }
  }
  return size % 2 == 0 ? (double) (nums[size / 2] + nums[size / 2 - 1]) / 2 : nums[size / 2];
}
```

#### [5. 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)

```c_cpp
string longestPalindrome(string s) {
  int n = s.size();
  //dp[i][j] 为 [i..j]是否为回文子串
  vector<vector<bool>> dp(n, vector<bool>(n));
  //每个字符本身就是回文串
  for (int i = 0; i < n; ++i) {
    dp[i][i] = true;
  }
  int len = 1, start = 0;
  for (int right = 1; right < n; ++right) {
    for (int left = 0; left < right; ++left) {
      if (s[left] == s[right]) {
        if (right - left + 1 < 3)//left==right且长度<3是一定是回文串
          dp[left][right] = true;
        else//长度>=3 则要看内部是否为回文串
          dp[left][right] = dp[left + 1][right - 1];

        if (dp[left][right]) {//找到回文子串，更新长度和起始点
          if (right - left + 1 > len) {
            len = right - left + 1;
            start = left;
          }
        }
      }
    }
  }
  return s.substr(start, len);
}
```

#### [10. 正则表达式匹配](https://leetcode-cn.com/problems/regular-expression-matching/)

```c_cpp
bool isMatch(string s, string p) {
    int m = s.size(), n = p.size();
    /*Dp[i][j] 代表 s[i]和p[j]是匹配*/
    vector<vector<bool>> dp(m + 1, vector<bool>(n + 1));/*多一行一列用于空字符的匹配*/
    dp[0][0] = true;/*空字符与空字符匹配*/
    /*第一行第一列为边界，第一列与空字符匹配都为false(vector默认false),第一列需要初始化*/
    for (int j = 2; j <= n; j += 2) {/*若p的偶数个为*则可以尝试与空字符匹配*/
        dp[0][j] = (p[j - 1] == '*' && dp[0][j - 2]);
    }
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            if (p[j - 1] == '*') {
                /*情况一，*重复0次，则看前两个字符位置是否匹配*/
                /*情况二，*重复前一个字符.则看前一个字符是否匹配，以及之前的和s[i-1]和p[j]是否匹配，若都匹配则可以复制前一个字符使用*/
                dp[i][j] = dp[i][j - 2] || ((s[i - 1] == p[j - 2] || p[j - 2] == '.') && dp[i - 1][j]);
            } else {
                /*当前字符不为*则看前一个字符是否匹配和当前字符是否匹配*/
                dp[i][j] = dp[i - 1][j - 1] && (s[i - 1] == p[j - 1] || p[j - 1] == '.');
            }
        }
    }
    return dp[m][n];
}
/*递归*/
bool isMatch1(string s, string p) {
    int len1 = s.size();
    int len2 = p.size();
    /*出口*/
    if (len2 == 0)
        return len1 == 0;
    /*判断第一个字符是否相等，若p[0]==‘。’ 直接相等*/
    bool firstMatch = (len1 > 0 && len2 > 0 && (s[0] == p[0] || p[0] == '.'));
    /*‘*’右两种情况*/
    if (len2 > 1 && p[1] == '*') {
        /*情况一：*号不产生作用，使得前一个字符重复0次*/
        /*情况二：*号产生作用，使得前一个字符重复,前提前一个字符需匹配*/
        return isMatch(s, p.substr(2)) || (firstMatch && isMatch(s.substr(1), p));
    }
    /*继续判断后面的是否匹配*/
    return firstMatch && isMatch(s.substr(1), p.substr(1));
}
```

#### [11.盛最多水的容器](https://leetcode-cn.com/problems/container-with-most-water/)

```c_cpp
int maxArea(vector<int> &height) {
  //双指针
  int left = 0, right = height.size() - 1;
  int ans = 0;
  while (left < right) {
    //宽
    int w = right - left;
    //盛水量由较小的高决定，并且移动较小的高
    int h = height[right] < height[left] ? height[right--] : height[left++];
    ans = max(ans, w * h);
  }
  return ans;
}
```

#### [15.三数之和](https://leetcode-cn.com/problems/3sum/)

```c_cpp
vector<vector<int>> threeSum(vector<int> &nums) {
  int n = nums.size();
  vector<vector<int>> ans;
  if (n < 3)
    return ans;
  sort(nums.begin(), nums.end());
  for (int i = 0; i < n - 2; ++i) {
    if (nums[i] > 0)//第一个数就大于0，加上其他两个数不可能为0直接退出
      break;
    //去重
    if (i > 0 && nums[i] == nums[i - 1])
      continue;
    //双指针
    int left = i + 1, right = n - 1;
    while (left < right) {
      int sum = nums[left] + nums[right] + nums[i];
      if (sum == 0) {
        ans.push_back({nums[i], nums[left], nums[right]});
        //去重
        while (left < right && nums[left] == nums[left + 1])
          ++left;
        while (left < right && nums[right] == nums[right - 1])
          --right;
        //更新指针
        ++left;
        --right;
      } else if (sum > 0) {
        --right;
      } else {
        ++left;
      }
    }
  }
  return ans;
}
```

#### [17.电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

```c_cpp
unordered_map<char, string> table = {
    {'2', "abc"},
    {'3', "def"},
    {'4', "ghi"},
    {'5', "jkl"},
    {'6', "mno"},
    {'7', "pqrs"},
    {'8', "tuv"},
    {'9', "wxyz"},
};
vector<string> ans;
string combination;
vector<string> letterCombinations(string digits) {
  if (digits.empty())
    return ans;
  backTrack(digits, 0);
  return ans;
}
void backTrack(const string &digits, int index) {
  int n = digits.size();
  if (index == n) {//退出条件
    ans.push_back(combination);
    return;
  }
  char digit = digits[index];
  string alphas = table[digit];
  for (const auto &item : alphas) {
    combination.push_back(item);
    backTrack(digits, index + 1);
    combination.pop_back();
  }
}
```

#### [19.删除链表的倒数第 N 个结点](https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/)

```c_cpp
ListNode *removeNthFromEnd(ListNode *head, int n) {
  auto *dummy = new ListNode(0, head);
  ListNode *first = head;
  //第二个指针指向dummy,方便删除操作，若指向head,则需要删除的指针为second
  ListNode *second = dummy;
  while (n--) {
    first = first->next;
  }
  while (first) {
    first = first->next;
    second = second->next;
  }
  second->next = second->next->next;
  //不能直接返回head结点，若删除的节点刚好是head
  ListNode *ans = dummy->next;
  delete dummy;
  return ans;
}
```

#### [20.有效的括号](https://leetcode-cn.com/problems/valid-parentheses/)

```c_cpp
  bool isValid(string s) {
    //栈
    stack<char> st;
    for (const auto &ch : s) {
      if (ch == '(' || ch == '[' || ch == '{')
        st.push(ch);
      else {
        if (st.empty() || (ch == ')' && st.top() != '(') || (ch == ']' && st.top() != '[')
            || (ch == '}' && st.top() != '{'))
          return false;
        st.pop();
      }
    }
    return st.empty();//遍历完后需要为空
  }
```

#### [21.合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```c_cpp
ListNode *mergeTwoLists(ListNode *list1, ListNode *list2) {
  auto dummy = new ListNode(-1);
  ListNode *p = dummy;
  while (list1 && list2) {
    if (list1->val <= list2->val) {
      p->next = list1;
      list1 = list1->next;
    } else {
      p->next = list2;
      list2 = list2->next;
    }
    p = p->next;
  }
  //判断剩余的链表
  p->next = list1 ? list1 : list2;
  ListNode *ans = dummy->next;
  delete dummy;
  return ans;
}
```

#### [22.括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

```c_cpp
vector<string> ans;
string combination;
vector<string> generateParenthesis(int n) {
  backTrack(n, 0, 0);
  return ans;
}
void backTrack(int n, int left, int right) {
  if (combination.size() == 2 * n)
    ans.push_back(combination);
  if (left < n) {
    combination.push_back('(');
    backTrack(n, left + 1, right);
    combination.pop_back();
  }
  if (right < left) {//右括号数量小于左括号
    combination.push_back(')');
    backTrack(n, left, right + 1);
    combination.pop_back();
  }
}
```

#### [23.合并两个有序链表](https://leetcode-cn.com/problems/merge-two-sorted-lists/)

```c_cpp
struct Status {
  int val;
  ListNode *ptr;
  bool operator<(const Status &rhs) const {
    return val > rhs.val;
  }
};
priority_queue<Status> priority_queue_;
ListNode *mergeKLists(vector<ListNode *> &lists) {
  //优先队列
  for (auto item : lists) {
    if (item)
      priority_queue_.push({item->val, item});
  }
  ListNode head, *tail = &head;
  while (!priority_queue_.empty()) {
    auto p = priority_queue_.top();
    priority_queue_.pop();
    tail->next = p.ptr;
    tail = tail->next;
    if (p.ptr->next) {
      priority_queue_.push({p.ptr->next->val, p.ptr->next});
    }
  }
  return head.next;
}

ListNode *mergeTwoList(ListNode *list1, ListNode *list2) {
  if (!list1 || !list2)
    return list1 ? list1 : list2;
  ListNode head, *p = &head;
  while (list1 && list2) {
    if (list1->val <= list2->val) {
      p->next = list1;
      list1 = list1->next;
    } else {
      p->next = list2;
      list2 = list2->next;
    }
    p = p->next;
  }
  p->next = list1 ? list1 : list2;
  return head.next;
}
ListNode *merge(vector<ListNode *> lists, int left, int right) {
  //分治合拼
  if (left == right)
    return lists[left];
  if (left > right)
    return nullptr;
  int mid = (right - left) / 2 + left;
  return mergeTwoList(merge(lists, left, mid), merge(lists, mid + 1, right));
}
ListNode *mergeKLists(vector<ListNode *> &lists) {
  //分治合并
  int n = lists.size();
  return merge(lists, 0, n - 1);
}

ListNode *mergeKLists(vector<ListNode *> &lists) {
  //顺序排序合拼
  if (lists.empty())
    return nullptr;
  int n = lists.size();
  ListNode *ans = nullptr;
  for (int i = 0; i < n; ++i) {
    ans = mergeTwoList(ans, lists[i]);
  }
  return ans;
}

```

#### [31.下一个排列](https://leetcode-cn.com/problems/next-permutation/)

```c_cpp
void nextPermutation(vector<int> &nums) {
  int n = nums.size();
  //从后往前找到第一个顺序对(i,i+1) 满足num[i]>=[i+1],此时较小的数为i,而[i+1,n)必然为降序
  int i = n - 2;
  while (i >= 0 && nums[i] >= nums[i + 1])
    --i;
  if (i >= 0) {
    //在[i+1,n)之间找到第一个大于num[i]的数，即较大数
    int j = n - 1;
    while (j > i && nums[i] > nums[j])
      --j;
    swap(nums[i], nums[j]);
  }
  //翻转[i+1,n)
  reverse(nums.begin() + i + 1, nums.end());
}
```

#### [32.最长有效括号](https://leetcode-cn.com/problems/longest-valid-parentheses/)DP

```c_cpp
int longestValidParentheses(string s) {
  int n = s.size();
  if (n <= 1)
    return 0;
  //定义dp[i]表示以i结尾的最长的括号长度
  vector<int> dp(n, 0);
  //边界条件
  dp[1] = (s[0] == '(' && s[1] == ')') ? 2 : 0;
  int ans = dp[1];
  for (int i = 2; i < n; ++i) {
    if (s[i] == ')') {
      if (s[i - 1] == '(')//当前括号长度+之前括号的长度
        dp[i] = dp[i - 2] + 2;
      else if (dp[i - 1] > 0) {//表示当前括号中含有括号
        if (i - dp[i - 1] - 1 >= 0 && s[i - dp[i - 1] - 1] == '(') {//表示内含的括号前一个位置和当前位置能够匹配
          dp[i] = dp[i - 1] + 2;
          if (i - dp[i - 1] - 2 >= 0)//加上前面的长度
            dp[i] += dp[i - dp[i - 1] - 2];
        }
      }
    }
    ans = max(ans, dp[i]);
  }
  return ans;
}
```

#### [33.搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

```c_cpp
int search(vector<int> &nums, int target) {
  //二分
  int n = nums.size();
  if (n == 0)
    return -1;
  if (n == 1)
    return nums[0] == target ? 0 : -1;
  int left = 0, right = n - 1;
  while (left <= right) {
    int mid = (right - left) / 2 + left;
    if (nums[mid] == target)
      return mid;
    if (nums[mid] >= nums[0]) {//当前mid在左段
      //判断target是否在(0，mid)
      if (nums[0] <= target && target < nums[mid])
        right = mid - 1;
      else
        left = mid + 1;
    } else {//当前mid在右段
      //判断target是否在(mid,n-1)
      if (nums[mid] < target && target <= nums[n - 1])
        left = mid + 1;
      else
        right = mid - 1;
    }
  }
  return -1;
}
```

#### [34.在排序数组中查找元素的第一个和最后一个位置](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

```c_cpp
vector<int> searchRange(vector<int> &nums, int target) {
  vector<int> ans = {-1, -1};
  if (nums.empty())
    return ans;
  int n = nums.size();
  int left = 0, right = n - 1;
  while (left < right) {
    int mid = (right - left) / 2 + left;
    if (target <= nums[mid])
      right = mid;
    else
      left = mid + 1;
  }
  if (nums[left] != target)
    return ans;
  ans[0] = left;
  left = 0, right = n - 1;
  while (left < right) {
    //注意当left=mid的时，需要向上取整，考左，不然会死循环
    int mid = (right - left + 1) / 2 + left;
    if (target >= nums[mid]) {
      left = mid;
    } else {
      right = mid - 1;
    }
  }
  ans[1] = left;
  return ans;
}
```

#### [39.组合总和](https://leetcode-cn.com/problems/combination-sum/)

```c_cpp
vector<vector<int>> ans;
vector<int> combination;
vector<vector<int>> combinationSum(vector<int> &candidates, int target) {
  backTrack(candidates, target, 0);
  return ans;
}

void backTrack(const vector<int> &candidates, int target, int index) {
  if (0 == target) {
    ans.push_back(combination);
    return;
  }
  if (target < 0)
    return;
  int n = candidates.size();
  for (int i = index; i < n; ++i) {
    combination.push_back(candidates[i]);
    backTrack(candidates, target - candidates[i], i);
    combination.pop_back();
  }
}
```

#### [42. 接雨水](https://leetcode-cn.com/problems/trapping-rain-water/)

```c_cpp
/*动态规划，提前计算左右的最高值*/
int trap(vector<int> &height) {
  int size = height.size();
  /*提前处理左右的最大值*/
  vector<int> left_max(size);
  left_max[0] = height[0];
  for (int i = 1; i < size; ++i) {
    left_max[i] = max(left_max[i - 1], height[i]);
  }
  vector<int> right_max(size);
  right_max[size - 1] = height[size - 1];
  for (int i = size - 2; i >= 0; --i) {
    right_max[i] = max(right_max[i + 1], height[i]);
  }

  int res = 0;
  for (int i = 0; i < size; ++i) {
    res += min(left_max[i], right_max[i]) - height[i];
  }
  return res;
}
/*暴力*/
int trap(vector<int> &height) {
  int size = height.size();
  int res = 0;
  for (int i = 0; i < size; ++i) {
    /*遍历获得当前位置的左右两侧的最高高度*/
    int leftMax = 0, rightMax = 0;
    for (int j = i - 1; j >= 0; --j) {
      leftMax = max(leftMax, height[j]);
    }
    for (int j = i + 1; j < size; ++j) {
      rightMax = max(rightMax, height[j]);
    }
    /*左右高度取最小然后减去当前高度获得当前位置可存储的水量*/
    if (min(leftMax, rightMax) > height[i])
      res += min(leftMax, rightMax) - height[i];
  }
  return res;
}
```

#### [46.全排列](https://leetcode-cn.com/problems/permutations/)

```c_cpp
vector<vector<int> > ans;
vector<int> combination;
vector<bool> used;
vector<vector<int>> permute(vector<int> &nums) {
  int n = nums.size();
  used.resize(n, false);
  backTrack(nums, used);
  return ans;
}
void backTrack(const vector<int> &nums, vector<bool> &used) {
  int n = nums.size();
  if (combination.size() == n) {
    ans.push_back(combination);
    return;
  }

  for (int i = 0; i < n; ++i) {
    if (used[i])
      continue;
    combination.push_back(nums[i]);
    used[i] = true;
    backTrack(nums, used);
    used[i] = false;
    combination.pop_back();
  }
}
```

#### [48.旋转图像](https://leetcode-cn.com/problems/rotate-image/)

```c_cpp
void rotate(vector<vector<int>> &matrix) {
  int n = matrix.size();
  //水平线镜像
  for (int i = 0; i < n / 2; ++i) { 
    for (int j = 0; j < n; ++j) {
      swap(matrix[i][j], matrix[n - i - 1][j]);
    }
  }
  //对角线翻转
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      swap(matrix[i][j], matrix[j][i]);
    }
  }
}
void rotate2(vector<vector<int>> &matrix) {
    int n = matrix.size();
    for (int i = 0; i < n / 2; ++i) {
        for (int j = 0; j < (n + 1) / 2; ++j) {
            int temp = matrix[i][j];
            matrix[i][j] = matrix[n - 1 - j][i];
            matrix[n - 1 - j][i] = matrix[n - 1 - i][n - 1 - j];
            matrix[n - 1 - i][n - 1 - j] = matrix[j][n - 1 - i];
            matrix[j][n - 1 - i] = temp;
        }
    }
}
```

#### [49.字母异位词分组](https://leetcode-cn.com/problems/group-anagrams/)

```c_cpp
vector<vector<string>> groupAnagrams(vector<string> &strs) {
  unordered_map<int, vector<string>> table;
  for (const auto &str : strs) {
    table[hash(str)].push_back(str);
  }
  vector<vector<string>> ans;
  for (const auto &item : table) {
    ans.push_back(item.second);
  }
  return ans;
}
int hash(const string &str) {
  int hash = 0;
  for (const auto &i : str) {
    hash += 5 * i * i * i * i / 26 + i * 955 + i * 996;
  }
  return hash;
}
```

#### [53.最大子数组和](https://leetcode-cn.com/problems/maximum-subarray/)

```c_cpp
int maxSubArray(vector<int> &nums) {
  int maxSub = nums[0];
  int p = nums[0];
  int n = nums.size();
  for (int i = 1; i < n; ++i) {
    int q = max(p, 0) + nums[i];
    maxSub = max(maxSub, q);
    p = q;
  }
  return maxSub;
}
```

#### [55.跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

```c_cpp
bool canJump(vector<int> &nums) {
  int reach = 0;//最大能跳跃的位置
  int n = nums.size();
  for (int i = 0; i < n; ++i) {
    if (i > reach)//若当前i大于最大能跳跃的位置,则为false
      return false;
    reach = max(reach, i + nums[i]);
  }
  return true;
}
```

#### [56.合并区间](https://leetcode-cn.com/problems/merge-intervals/)

```c_cpp
vector<vector<int>> merge(vector<vector<int>> &intervals) {
  vector<vector<int>> ans;
  int n = intervals.size();
  if (n == 0)
    return ans;
  std::sort(intervals.begin(), intervals.end());//排序
  for (int i = 0; i < n; ++i) {//使用ans的最后一个数组，与当前数组，进行比较是否需要进行合并
    int left = intervals[i][0], right = intervals[i][1];
    if (ans.empty() || ans.back()[1] < left) {//不需要进行合并
      ans.push_back({left, right});
    } else {//合并，取右侧最大值
      ans.back()[1] = max(ans.back()[1], right);
    }
  }
  return ans;
}
```

#### [62.不同路径](https://leetcode-cn.com/problems/unique-paths/)

```c_cpp
int uniquePaths(int m, int n) {
  //dp[i][j]表示抵达当前位置路径数量
  vector<vector<int>> dp(m, vector<int>(n));
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (i == 0 || j == 0) {//边界条件
        dp[i][j] = 1;
      } else {
        dp[i][j] = dp[i - 1][j] + dp[i][j - 1];//从上和左抵达当前位置
      }
    }
  }
  return dp[m - 1][n - 1];
}
```

#### [64.最小路径和](https://leetcode-cn.com/problems/minimum-path-sum/)

```c_cpp
int minPathSum(vector<vector<int>> &grid) {
  int m = grid.size(), n = grid[0].size();
  //dp[i][j]表示抵达当前位置最小和
  vector<vector<int>> dp(m, vector<int>(n));
  //边界条件
  dp[0][0] = grid[0][0];
  for (int i = 1; i < m; ++i) {
    dp[i][0] = dp[i - 1][0] + grid[i][0];
  }
  for (int j = 1; j < n; ++j) {
    dp[0][j] = dp[0][j - 1] + grid[0][j];
  }

  for (int i = 1; i < m; ++i) {
    for (int j = 1; j < n; ++j) {
      dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];//从上和左中选最小值 + 当前位置的值
    }
  }

  return dp[m - 1][n - 1];
}
```

#### [70.爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

```c_cpp
//内存优化，只需要前两阶的值
int climbStairs(int n) {
  if (n == 1 || n == 2) {
    return n == 1 ? 1 : 2;
  }
  int a = 1, b = 2, c;
  for (int i = 3; i <= n; ++i) {
    c = a + b;
    tie(a, b) = {b, c};
  }
  return c;
}
int climbStairs1(int n) {
  vector<int> dp(n + 1);
  dp[0] = 0;
  dp[1] = 1;
  dp[2] = 2;
  for (int i = 3; i <= n; ++i) {
    //可以从前一阶，也可从前两节抵达当前阶梯
    dp[i] = dp[i - 1] + dp[i - 2];
  }
  return dp[n];
}
```

#### [72.编辑距离](https://leetcode-cn.com/problems/edit-distance/)

```c_cpp
int minDistance(string word1, string word2) {
  int m = word1.size(), n = word2.size();
  //dp[i][j] 表示 word[i] 变成word2[j]所需要编辑的次数
  vector<vector<int>> dp(m + 1, vector<int>(n + 1));
  //边界条件
  for (int i = 1; i <= m; ++i) {//空字符到word1所需编辑的次数
    dp[i][0] = i;
  }
  for (int j = 1; j <= n; ++j) {//空字符到word2所需要编辑的距离
    dp[0][j] = j;
  }
  for (int i = 1; i <= m; ++i) {
    for (int j = 1; j <= n; ++j) {
      if (word1[i - 1] == word2[j - 1]) {
        dp[i][j] = dp[i - 1][j - 1];//当前字符相同，则直接从左上角的值
      } else {
        dp[i][j] = min(dp[i - 1][j], min(dp[i][j - 1], dp[i - 1][j - 1])) + 1;//在最小的操作上+1；
      }
    }
  }
  return dp[m][n];
}
```

#### [75.颜色分类](https://leetcode-cn.com/problems/sort-colors/)

```c_cpp
void sortColors(vector<int> &nums) {
  int n = nums.size();
  int p0 = 0, p2 = n - 1;
  for (int i = 0; i <= p2; ++i) {
    while (i < p2 && nums[i] == 2) {//将2放到最后面,考虑缓过来的还是2，使用while直到不是2
      swap(nums[i], nums[p2--]);
    }

    if (nums[i] == 0) {
      swap(nums[i], nums[p0++]);
    }
  }
}
```

#### [76.最小覆盖子串](https://leetcode-cn.com/problems/minimum-window-substring/)

```c_cpp
unordered_map<char, int> cnt, ori;
bool check() {
  //判断记录的字符数量是否相等
  for (const auto &p : ori) {
    if (cnt[p.first] < p.second)
      return false;
  }
  return true;
}
string minWindow(string s, string t) {
  for (const auto &ch : t) {
    ++ori[ch];
  }
  int l = 0, r = -1, n = s.size();
  int len = INT_MAX, start = -1;
  while (r < n) {
    //判断当前字符是否是所需要的，如果是则使用cnt记录
    if (ori.find(s[++r]) != ori.end()) {
      ++cnt[s[r]];
    }
    //检查是否满足条件
    while (check() && l <= r) {
      //更新最小值
      if (r - l + 1 < len) {
        len = r - l + 1;
        start = l;
      }
      //移动l缩小窗口
      if (ori.find(s[l]) != ori.end()) {
        --cnt[s[l]];
      }
      ++l;
    }
  }
  return start == -1 ? string() : s.substr(start, len);
}
```

#### [78.子集](https://leetcode-cn.com/problems/subsets/)

```c_cpp
vector<vector<int>> subsets1(vector<int> &nums) {
  vector<vector<int>> ans;
  ans.push_back(vector<int>());
  int n = nums.size();
  for (int i = 0; i < n; ++i) {
    int size = ans.size();
    for (int j = 0; j < size; ++j) {
      auto cur = ans[j];
      cur.push_back(nums[i]);
      ans.push_back(cur);
    }
  }
  return ans;
}

//回溯
vector<vector<int>> ans;
vector<int> combination;
vector<vector<int>> subsets(vector<int> &nums) {
  backTrack(nums, 0);
  return ans;
}

void backTrack(const vector<int> &nums, int index) {
  ans.push_back(combination);//所有子集都记录
  int n = nums.size();
  for (int i = index; i < n; ++i) {
    combination.push_back(nums[i]);
    backTrack(nums, i + 1);
    combination.pop_back();
  }
}
```

#### [79.单词搜索](https://leetcode-cn.com/problems/word-search/)

```c_cpp
bool exist(vector<vector<char>> &board, string word) {
  int m = board.size(), n = board[0].size();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      if (board[i][j] == word[0]) {
        //回溯
        if (backTrack(board, word, i, j, 0))
          return true;
      }
    }
  }
  return false;
}
int dirs[4][2] = {{-1, 0}, {1, 0}, {0, 1}, {0, -1},};
bool backTrack(vector<vector<char>> &board, const string &word, int x, int y, int index) {
  if (board[x][y] != word[index])//不符合
    return false;
  if (index == word.size() - 1)//匹配到最后一个字符
    return true;
  int m = board.size(), n = board[0].size();
  char origin = board[x][y];
  board[x][y] = '.';
  for (auto &dir : dirs) {
    int nx = x + dir[0], ny = y + dir[1];
    //越界,已使用
    if (nx < 0 || nx >= m || ny < 0 || ny >= n || board[nx][ny] == '.')
      continue;
    if (backTrack(board, word, nx, ny, index + 1))//四个方向任意一个为true,则为true
      return true;
  }
  board[x][y] = origin;//回溯
  return false;
}
```

#### [84.柱状图中最大的矩形](https://leetcode-cn.com/problems/largest-rectangle-in-histogram/)

```c_cpp
int largestRectangleArea(vector<int> &heights) {
  if (heights.empty())
    return 0;
  if (heights.size() == 1)
    return heights[0];
  //头尾插入哨兵
  heights.insert(heights.begin(), 0);
  heights.push_back(0);
  int n = heights.size();
  int ans = 0;
  stack<int> st;
  st.push(0);//插入哨兵位置
  for (int i = 1; i < n; i++) {
    //保持栈单调递增，若出现下降，则处理栈中的数据,计算面积
    while (!st.empty() && heights[st.top()] > heights[i]) {
      int height = heights[st.top()];
      st.pop();
      int width = i - st.top() - 1;
      ans = max(ans, width * height);
    }
    st.push(i);
  }
  return ans;
}
```

#### [85.最大矩形](https://leetcode-cn.com/problems/maximal-rectangle/)

```c_cpp
int maximalRectangle(vector<vector<char>> &matrix) {
  int m = matrix.size(), n = matrix[0].size();
  //预处理，获得每行连续的1的长度
  vector<vector<int>> left(m, vector<int>(n));
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (matrix[i][j] == '1') {
        left[i][j] = (j > 0 ? left[i][j - 1] : 0) + 1;
      }
    }
  }
  int maxArea = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (matrix[i][j] == '1') {
        int width = left[i][j];
        int area = width;
        for (int k = i - 1; k >= 0; k--) {//从当前位置向向上走，取最小宽度，计算面积
          width = min(width, left[k][j]);
          area = max(area, width * (i - k + 1));
        }
        maxArea = max(maxArea, area);
      }
    }
  }
  return maxArea;
}
```

#### [94.二叉树的中序遍历](https://leetcode-cn.com/problems/binary-tree-inorder-traversal/)

```c_cpp
vector<int> inorderTraversal(TreeNode *root) {
  //迭代
  vector<int> ret;
  stack<TreeNode *> st;
  auto p = root;
  while (p != nullptr || !st.empty()) {
    //不断往左走，直到空
    while (p != nullptr) {
      st.push(p);
      p = p->left;
    }
    //取出父节点
    p = st.top();
    st.pop();
    ret.push_back(p->val);
    //往右走
    p = p->right;
  }
  return ret;
}
vector<int> inorderTraversal1(TreeNode *root) {
  vector<int> ret;
  DFS(root, ret);
  return ret;
}
void DFS(TreeNode *node, vector<int> &ans) {
  if (!node)
    return;
  DFS(node->left, ans);
  ans.push_back(node->val);
  DFS(node->right, ans);
}
```

#### [96.不同的二叉搜索树](https://leetcode-cn.com/problems/unique-binary-search-trees/)

```c_cpp
int numTrees(int n) {
//    大问题中找到子问题：求G(n)，即求[1,n]的解的和，那就先以其中的i(1<=i<=n)为顶点, 记为f(i)
//    解决子问题：i为顶点的解，为G[i-1] * G[n-i]的解。
//    合并子问题求的大问题的解: G[n] = f(1) +....+f(i) + ...f(n)
  vector<int> dp(n + 1);
  dp[0] = 1;//空树
  dp[1] = 1;//一个节点只有一种树；
  for (int i = 2; i <= n; ++i) {
    for (int j = 1; j <= i; ++j) {//以i为根节点的树种类
      dp[i] += dp[j - 1] * dp[i - j];
    }
  }
  return dp[n];
}
```

#### [98.验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

```c_cpp
bool isValidBST(TreeNode *root) {
  //迭代
  long pre = LONG_MIN;
  stack<TreeNode *> st;
  auto p = root;
  while (p != nullptr || !st.empty()) {
    //往左走
    while (p != nullptr) {
      st.push(p);
      p = p->left;
    }
    //取出父节点
    p = st.top();
    st.pop();
    if (p->val <= pre)
      return false;
    pre = p->val;//更新前继节点值
    //往右走
    p = p->right;
  }
  return true;
}
bool isValidBST1(TreeNode *root) {
  return DFS(root);
}
long pre = LONG_MIN;
bool DFS(TreeNode *node) {//中序遍历
  if (node == nullptr)
    return true;
  //左子树
  bool left = DFS(node->left);
  //若当前节点小于前继节点则为false
  if (node->val <= pre)
    return false;
  pre = node->val;//更新前继节点值
  //右子树
  bool right = DFS(node->right);
  //返回左右子树是否符合
  return left && right;
}
```

#### [101.对称二叉树](https://leetcode-cn.com/problems/symmetric-tree/)

```c_cpp
bool isSymmetric(TreeNode *root) {
  return DFS(root->left, root->right);
}
bool DFS(TreeNode *node1, TreeNode *node2) {
  if (!node1 && !node2)
    return true;
  else if (!node1 || !node2)
    return false;
  else if (node1->val != node2->val)
    return false;
  else
    return DFS(node1->left, node2->right) && DFS(node1->right, node2->left);
}
```

#### [102.二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

```c_cpp
vector<vector<int>> levelOrder(TreeNode *root) {
  if (root == nullptr)
    return {};
  queue<TreeNode *> queue;
  queue.push(root);
  vector<vector<int>> ret;
  while (!queue.empty()) {
    int size = queue.size();
    TreeNode *p;
    vector<int> temp;
    for (int i = 0; i < size; i++) {
      p = queue.front();
      queue.pop();
      temp.push_back(p->val);
      if (p->left)
        queue.push(p->left);
      if (p->right)
        queue.push(p->right);
    }
    ret.emplace_back(temp);
  }
  return ret;
}
```

#### [104.二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

```c_cpp
int maxDepth(TreeNode *root) {
  return DFS(root);
}
int DFS(TreeNode *root) {
  if (root == nullptr)
    return 0;
  return max(DFS(root->left), DFS(root->right)) + 1;
}
```

#### [105.从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

```c_cpp
TreeNode *buildTree(vector<int> &preorder, vector<int> &inorder) {
  int n = inorder.size();
  //用于加速查询根节点在中序遍历的位置
  unordered_map<int, int> map;
  for (int i = 0; i < n; i++) {
    map[inorder[i]] = i;
  }
  return DFS(preorder, inorder, map, 0, n - 1, 0, n - 1);
}
TreeNode *DFS(const vector<int> &pre,
              const vector<int> &in,
              const unordered_map<int, int> &map,
              int preL,
              int preR,
              int inL,
              int inR) {
  if (preL > preR || inL > inR)
    return nullptr;
  auto *root = new TreeNode(pre[preL]);
  int rootIndex = map.at(root->val);//根节点在中序遍历的位置
  //通过rootIndex,记录左右子树的节点数量
  root->left = DFS(pre, in, map, preL + 1, rootIndex - inL + preL, inL, rootIndex - 1);
  root->right = DFS(pre, in, map, rootIndex - inL + preL + 1, preR, rootIndex + 1, inR);
  return root;
}
```

#### [114.二叉树展开为链表](https://leetcode-cn.com/problems/flatten-binary-tree-to-linked-list/)

```c_cpp
void flatten1(TreeNode *root) {
  /*思路
   * 1. 交换当前节点左右子树
   * 2. 找到当前节点最右边的子树
   * 3. 将当前节点左子树挂到最右边的子树
   * 4 递归当前节点的右子树
   * */
  if (root == nullptr)
    return;
  swap(root->left, root->right);
  auto curr = root;
  while (curr->right) {
    curr = curr->right;
  }
  curr->right = root->left;
  root->left = nullptr;
  flatten1(root->right);
}

void flatten(TreeNode *root) {
  //暴力，使用先序遍历将节点存储到数组中，然后重新链接
  vector<TreeNode *> nodes;
  DFS(root, nodes);
  int n = nodes.size();
  for (int i = 1; i < n; i++) {
    nodes[i - 1]->right = nodes[i];
    nodes[i - 1]->left = nullptr;
  }
}
void DFS(TreeNode *node, vector<TreeNode *> &nodes) {
  if (node == nullptr)
    return;
  nodes.push_back(node);
  DFS(node->left, nodes);
  DFS(node->right, nodes);
}
```

#### [121.买卖股票的最佳时机](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  //dp[i][0] 代表在第i天买入时的最大利润，dp[i][1]代表在第i天卖出时的最大利润
  vector<vector<int>> dp(n, vector<int>(2));
  //边界条件，第一天买入以及第一天卖出为0
  dp[0][0] = -prices[0];
  dp[0][1] = 0;
  int maxProfit = 0;
  for (int i = 1; i < n; ++i) {
    dp[i][0] = max(dp[i - 1][0], -prices[i]);//买入时的最大利润为，max(昨天买入，今天买入)
    dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] + prices[i]);//卖出时的最大利润为，max(昨天卖出，昨天持有+今天股价)
  }
  return dp[n - 1][1];
}
```

#### [124.二叉树中的最大路径和](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/)

```c_cpp
int maxPath = INT_MIN;
int maxPathSum(TreeNode *root) {
  postOrder(root);
  return maxPath;
}
//后序遍历，先处理子树在处理本身本身
int postOrder(TreeNode *node) {
  if (node == nullptr)
    return 0;
  //舍去负数
  int left = max(postOrder(node->left), 0);
  int right = max(postOrder(node->right), 0);
  //更新最大值，最大路径可以包含本身
  maxPath = max(maxPath, node->val + left + right);
  return node->val + max(left, right);
}
```

#### [128.最长连续序列](https://leetcode-cn.com/problems/longest-consecutive-sequence/)

```c_cpp
int longestConsecutive(vector<int> &nums) {
  unordered_set<int> set(nums.begin(), nums.end());//去重
  int ret = 0;
  for (const auto &item: set) {
    if (!set.count(item - 1)) {//找不到比自己小的，然后在找比自己大的
      int curNum = item, curLen = 1;
      while (set.count(curNum + 1)) {//循环找比item大的连续的数
        ++curLen;
        ++curNum;
      }
      ret = max(ret, curLen);
    }
  }
  return ret;
}
```

#### [136.只出现一次的数字](https://leetcode-cn.com/problems/single-number/)

```c_cpp
int singleNumber(vector<int> &nums) {
  //0与任何数都为任何数
  //任何数与自己都为0
  int ret = 0;
  for (const auto &item: nums) {
    ret ^= item;
  }
  return ret;
}
```

#### [139.单词拆分](https://leetcode-cn.com/problems/word-break/)

```c_cpp
bool wordBreak(string s, vector<string> &wordDict) {
  int n = s.size();
  unordered_set<string> dict;
  int maxLen = 0;
  for (const auto &item: wordDict) {
    dict.insert(item);
    maxLen = max(maxLen, (int) item.size());
  }
  //dp[j]为分割点
  //枚举s[0...i-1]之间的中间点j,看s[0...j-1] 以及 s[j...i-j]是否合法
  vector<bool> dp(n + 1);
  dp[0] = true;//空字符
  for (int i = 1; i <= n; i++) {
    for (int j = max(0, i - maxLen); j < i; j++) {
      // 前一个分割点为true，且后面的未存在则当前i也为true
      if (dp[j] == true && dict.find(s.substr(j, i - j)) != dict.end()) {
        dp[i] = true;
        break;
      }
    }
  }
  return dp[n];
}
```

#### [141.环形链表](https://leetcode-cn.com/problems/linked-list-cycle/)

```c_cpp
bool hasCycle(ListNode *head) {
  //空节点或一个结点没有环
  if (head == nullptr || head->next == nullptr)
    return false;

  auto slow = head, fast = head->next;
  while (slow != fast) {
    //fast抵达空或下一个位置为空，则没有环
    if (fast == nullptr || fast->next == nullptr)
      return false;
    slow = slow->next;
    fast = fast->next->next;
  }
  return true;
}
```

#### [142.环形链表 II](https://leetcode-cn.com/problems/linked-list-cycle-ii/)

```c_cpp
ListNode *detectCycle(ListNode *head) {
  /* 推导过程
  slow * 2 = fast;
  slow = a + b;
  fast = a + b + c + b = a + 2*b + c;
  (a + b)*2 = a + 2*b + c;
  a = c;
  */
  if (head == nullptr || head->next == nullptr)//空指针或一个结点
    return nullptr;
  auto slow = head, fast = head;
  while (fast != nullptr && fast->next != nullptr) {
    slow = slow->next;
    fast = fast->next->next;
    if (slow == fast) {//相遇则说明有环
      //相遇后，在头结点新建指针，与slow指针同步向后走，直到ptr与slow相遇，即环的入口点
      auto ptr = head;
      while (ptr != slow) {
        slow = slow->next;
        ptr = ptr->next;
      }
      return ptr;
    }
  }
  return nullptr;
}
```

#### [146LRU 缓存](https://leetcode-cn.com/problems/lru-cache/)

```c_cpp
struct DLinkNode {
  int key, val;
  DLinkNode *prev, *next;
  DLinkNode() : key(0), val(0), prev(nullptr), next(nullptr) {}
  DLinkNode(int key, int value) : key(key), val(value), prev(nullptr), next(nullptr) {}
};
class LRUCache {
 private:
  int capacity;
  int size;
  DLinkNode *head, *tail;
  unordered_map<int, DLinkNode *> cache;
 public:
  LRUCache(int capacity) : capacity(capacity), size(0) {
    head = new DLinkNode();
    tail = new DLinkNode();
    head->next = tail;
    tail->prev = head;
  }

  int get(int key) {
    int ret = -1;
    if (cache.count(key)) {
      auto node = cache[key];
      moveToHead(node);
      ret = node->val;
    }
    return ret;
  }

  void put(int key, int value) {
    if (cache.count(key)) {
      auto node = cache[key];
      node->val = value;
      moveToHead(node);
    } else {
      auto node = new DLinkNode(key, value);
      cache[key] = node;
      insertToHead(node);
      if (++this->size > this->capacity) {//超出容量删除最后一个结点
        removeTail();
        --this->size;
      }
    }
  }

 private:
  void insertToHead(DLinkNode *node) {
    node->prev = head;
    node->next = head->next;
    head->next->prev = node;
    head->next = node;
  }
  void removeNode(DLinkNode *node) {
    node->prev->next = node->next;
    node->next->prev = node->prev;
  }
  void moveToHead(DLinkNode *node) {
    removeNode(node);
    insertToHead(node);
  }
  void removeTail() {
    auto node = tail->prev;
    cache.erase(node->key);
    removeNode(node);
    delete node;
  }
};
```

#### [148.排序链表](https://leetcode-cn.com/problems/sort-list/)

```c_cpp
/**
 * 合并两个链表，升序
 * @param head1
 * @param head2
 * @return
 */
ListNode *mergeTwoList(ListNode *head1, ListNode *head2) {//合并两个链表
  ListNode dummy;
  ListNode *p = &dummy, *temp1 = head1, *temp2 = head2;
  while (temp1 && temp2) {
    if (temp1->val <= temp2->val) {
      p->next = temp1;
      temp1 = temp1->next;
    } else {
      p->next = temp2;
      temp2 = temp2->next;
    }
    p = p->next;
  }
  p->next = temp1 == nullptr ? temp2 : temp1;
  return dummy.next;
}
/**
 * 归并排序
 * @param head
 * @param tail
 * @return
 */
ListNode *mergeSort(ListNode *head, ListNode *tail) {
  if (head->next == tail) {//只有一个结点，断开后直接返回
    head->next = nullptr;
    return head;
  }
  auto slow = head, fast = head;
  while (fast != tail) {
    slow = slow->next;
    fast = fast->next;
    if (fast != tail)
      fast = fast->next;
  }
  auto mid = slow;
  return mergeTwoList(mergeSort(head, mid), mergeSort(mid, tail));
}
ListNode *sortList(ListNode *head) {
  //归并排序
  if (head == nullptr)
    return head;
  return mergeSort(head, nullptr);
}
```

#### [152.乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

```c_cpp
int maxProduct(vector<int> &nums) {
  int n = nums.size();
  //记录最大值，记录最小值，应为负数的关系
  int maxNum = nums[0], minNum = nums[0], ans = nums[0];
  for (int i = 1; i < n; i++) {
    int mx = maxNum, mn = minNum;
    maxNum = max(mx * nums[i], max(mn * nums[i], nums[i]));
    minNum = min(mn * nums[i], min(mx * nums[i], nums[i]));
    ans = max(ans, maxNum);
  }
  return ans;
}
```

#### [155.最小栈](https://leetcode-cn.com/problems/min-stack/)

```c_cpp
class MinStack {
private:
stack<int> st;
//使用一个栈保持当前的最小值
stack<int> minSt;
public:
MinStack() {
  minSt.push(INT_MAX);
}

void push(int val) {
  st.push(val);
  if (val < minSt.top()) {
    minSt.push(val);
  } else {
    minSt.push(minSt.top());
  }
}

void pop() {
  st.pop();
  minSt.pop();
}

int top() {
  return st.top();
}

int getMin() {
  return minSt.top();
}
};
```

#### [160.相交链表](https://leetcode-cn.com/problems/intersection-of-two-linked-lists/)

```c_cpp
ListNode *getIntersectionNode1(ListNode *headA, ListNode *headB) {
  unordered_set<ListNode *> uset;
  while (headA) {
    uset.insert(headA);
    headA = headA->next;
  }
  while (headB) {
    if (uset.count(headB)) {
      return headB;
    }
    headB = headB->next;
  }
  return nullptr;
}
ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
  if (headA == nullptr || headB == nullptr)
    return nullptr;
  auto p = headA, q = headB;
  while (p != q) {
    p = p == nullptr ? headB : p->next;
    q = q == nullptr ? headA : q->next;
  }
  return p;
}
```

#### [169.多数元素](https://leetcode-cn.com/problems/majority-element/) 摩尔投票

```c_cpp
int majorityElement(vector<int> &nums) {
  //摩尔投票算法
  int n = nums.size();
  int ret = nums[0], cnt = 1;
  for (int i = 1; i < n; i++) {
    if (ret == nums[i]) {//出现相同的数，则投票
      ++cnt;
    } else {//出现不同的数，则减去当前数字的票数，若票数为0，则更换投票的数
      if (cnt > 0) {
        --cnt;
      } else {
        ret = nums[i];
        cnt = 1;
      }
    }
  }
  return ret;
}
```

#### [198.打家劫舍](https://leetcode-cn.com/problems/house-robber/)

```c_cpp
int rob(vector<int> &nums) {
  int n = nums.size();
  if (n == 1)
    return nums[0];
  if (n == 2)
    return max(nums[0], nums[1]);

  int a = nums[0];
  int b = max(nums[0], nums[1]);
  int c;
  for (int i = 2; i < n; i++) {
    c = max(nums[i] + a, b);
    tie(a, b) = {b, c};
  }
  return c;
}
```

#### [200.岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

```c_cpp
class UnionFind {
 private:
  int size;
  vector<int> parent;
 public:
  explicit UnionFind(int n) : size(n) {
    parent.resize(n);
    iota(parent.begin(), parent.end(), 0);
  }
  int Find(int x) {
    return x == parent[x] ? x : (parent[x] = Find(parent[x]));
  }
  void Union(int x, int y) {
    int acX = Find(x);
    int acY = Find(y);

    if (acX == acY)//同一个集合直接返回
      return;
    else { //不同集合，将y合并到x中,每次合并size-1,
      parent[acY] = acX;
      --this->size;
    }
  }
  int getSize() const {
    return this->size;
  }
};
class Solution {
 public:
  int numIslands(vector<vector<char>> &grid) {
    int m = grid.size(), n = grid[0].size();
    UnionFind union_find(m * n);
    int ocean = 0;
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (grid[i][j] == '0') {//记录海洋数量
          ++ocean;
        } else {
          //只需找下和右边是否连通
          if (i + 1 < m && grid[i + 1][j] == '1') {
            union_find.Union(i * n + j, (i + 1) * n + j);
          }
          if (j + 1 < n && grid[i][j + 1] == '1') {
            union_find.Union(i * n + j, i * n + (j + 1));
          }
        }
      }
    }
    return union_find.getSize() - ocean;
  }
```

#### [206.反转链表](https://leetcode-cn.com/problems/reverse-linked-list/)

```c_cpp
ListNode *reverseList(ListNode *head) {
  ListNode *pre = nullptr, *curr = head;
  while (curr != nullptr) {
    auto temp = curr->next;
    curr->next = pre;
    pre = curr;
    curr = temp;
  }
  return pre;
}
```

#### [207.课程表](https://leetcode-cn.com/problems/course-schedule/)

```c_cpp
bool canFinish(int numCourses, vector<vector<int>> &prerequisites) {
  //图 BFD
  vector<int> inDegree(numCourses);
  unordered_map<int, vector<int>> edges;//key前置课程，value完成前置后可以学习的课程
  for (const auto &item: prerequisites) {
    ++inDegree[item[0]];
    edges[item[1]].push_back(item[0]);
  }
  queue<int> queue;
  //将可以直接学习的课程（入度为0）的入队
  for (int i = 0; i < inDegree.size(); i++) {
    if (inDegree[i] == 0) {
      queue.push(i);
    }
  }
  //BFS
  int finishCourse = 0;
  while (!queue.empty()) {
    auto course = queue.front();
    queue.pop();
    ++finishCourse;
    for (const auto &relationCourse: edges[course]) {//前置课程完成，相关课程的入度-1
      if (--inDegree[relationCourse] == 0) {
        queue.push(relationCourse);
      }
    }
  }
  return finishCourse == numCourses;
}
```

#### [208.实现 Trie (前缀树)](https://leetcode-cn.com/problems/implement-trie-prefix-tree/)

```c_cpp
class Trie {
 private:
  bool isEnd;
  vector<Trie *> children;
 public:
  Trie() : isEnd(false), children(vector<Trie *>(26, nullptr)) {

  }

  void insert(string word) {
    auto node = this;
    for (const auto &ch: word) {
      int index = ch - 'a';
      if (node->children[index] == nullptr) {
        node->children[index] = new Trie();
      }
      node = node->children[index];
    }
    node->isEnd = true;
  }

  bool search(string word) {
    auto node = this;
    for (const auto &ch: word) {
      int index = ch - 'a';
      if (node->children[index] == nullptr)
        return false;
      node = node->children[index];
    }
    return node->isEnd;
  }

  bool startsWith(string prefix) {
    auto node = this;
    for (const auto &ch: prefix) {
      int index = ch - 'a';
      if (node->children[index] == nullptr)
        return false;
      node = node->children[index];
    }
    return true;
  }
};
```

#### [215.数组中的第K个最大元素](https://leetcode-cn.com/problems/kth-largest-element-in-an-array/)

```c_cpp
int partition(vector<int> &nums, int left, int right) {
  //[left,right]中任选一个作为轴值
  int aix = rand() % (right - left + 1) + left;
  //轴尾互换
  swap(nums[aix], nums[right]);
  //获得轴值x
  int x = nums[right], i = left - 1;
  for (int j = left; j < right; j++) {
    if (nums[j] <= x) {
      swap(nums[++i], nums[j]);
    }
  }
  swap(nums[++i], nums[right]);
  return i;
}
int quickSort(vector<int> &nums, int left, int right, int index) {
  int q = partition(nums, left, right);//任选一个轴值排序后，若恰好是所需要的位置，则直接返回
  if (q == index) {
    return nums[q];
  } else {
    return q < index ? quickSort(nums, q + 1, right, index) : quickSort(nums, left, q - 1, index);
  }
}
int findKthLargest(vector<int> &nums, int k) {
  int n = nums.size();
  int index = n - k;
  srand(time(0));
  return quickSort(nums, 0, n - 1, index);
}

int findKthLargest1(vector<int> &nums, int k) {
  priority_queue<int, vector<int>, greater<>> pq;
  int n = nums.size();
  for (int i = 0; i < n; i++) {
    if (pq.empty() || pq.size() < k || pq.top() < nums[i])
      pq.push(nums[i]);
    if (pq.size() > k) {
      pq.pop();
    }
  }
  return pq.top();
}
```

#### [221.最大正方形](https://leetcode-cn.com/problems/maximal-square/)

```c_cpp
int maximalSquare(vector<vector<char>> &matrix) {
  int m = matrix.size(), n = matrix[0].size();
  vector<vector<int>> dp(m, vector<int>(n));
  int maxEdge = 0;
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      if (matrix[i][j] == '1') {
        if (i == 0 || j == 0)
          dp[i][j] = 1;
        else
          dp[i][j] = min(dp[i - 1][j - 1], min(dp[i - 1][j], dp[i][j - 1])) + 1;
        maxEdge = max(maxEdge, dp[i][j]);
      }
    }
  }
  return maxEdge * maxEdge;
}
```

#### [226.翻转二叉树](https://leetcode-cn.com/problems/invert-binary-tree/)

```c_cpp
TreeNode *invertTree(TreeNode *root) {
  DFS(root);
  return root;
}
void DFS(TreeNode *node) {
  if (node == nullptr)
    return;
  swap(node->left, node->right);
  DFS(node->left);
  DFS(node->right);
}
```

#### [234.回文链表](https://leetcode-cn.com/problems/palindrome-linked-list/)

```c_cpp
ListNode *reverseList(ListNode *head) {
  ListNode *pre = nullptr, *curr = head;
  while (curr) {
    auto temp = curr->next;
    curr->next = pre;
    pre = curr;
    curr = temp;
  }
  return pre;
}
bool isPalindrome(ListNode *head) {
  if (head == nullptr)
    return false;
  //快慢指针找到中间节点
  //以中间节点为界限，翻转后半部分链表
  auto slow = head, fast = head;
  while (fast->next != nullptr && fast->next->next != nullptr) {
    slow = slow->next;
    fast = fast->next->next;
  }
  ListNode *p1 = head;
  ListNode *p2 = reverseList(slow->next);
  bool result = true;
  while (result && p2 != nullptr) {
    if (p1->val != p2->val)
      result = false;
    p1 = p1->next;
    p2 = p2->next;
  }
  return result;
}
bool isPalindrome1(ListNode *head) {
  //暴力，直接存储到vector中
  vector<ListNode *> listnode;
  while (head) {
    listnode.push_back(head);
    head = head->next;
  }
  int n = listnode.size();
  int left = 0, right = n - 1;
  while (left < right) {
    if (listnode[left++]->val != listnode[right--]->val)
      return false;
  }
  return true;
}
```

#### [236.二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

```c_cpp
TreeNode *lowestCommonAncestor(TreeNode *root, TreeNode *p, TreeNode *q) {
  if (root == nullptr)
    return nullptr;
  //当前节点为是其中一个结点，那么当前节点就是那个节点最近的祖先
  if (root->val == p->val || root->val == q->val)
    return root;
  auto left = lowestCommonAncestor(root->left, p, q);
  auto right = lowestCommonAncestor(root->right, p, q);
  //左右都不为空，说明找到祖先节点，那么当前节点为公共最近祖先节点
  if (left != nullptr && right != nullptr)
    return root;
  //左树为空，只能在右数
  if (left == nullptr)
    return right;
  //右树为空，只能是左树
  if (right == nullptr)
    return left;
  return nullptr;
}
```

#### [238.除自身以外数组的乘积](https://leetcode-cn.com/problems/product-of-array-except-self/)

```c_cpp
vector<int> productExceptSelf(vector<int> &nums) {
  int n = nums.size();
  //记录当前位置之前的乘积
  vector<int> left(n);
  left[0] = 1;
  for (int i = 1; i < n; i++) {
    left[i] = left[i - 1] * nums[i - 1];
  }
  int r = 1;
  for (int i = n - 1; i >= 0; i--) {
    //当前位置左侧*右侧的乘积
    left[i] *= r;
    //更新右侧乘积
    r *= nums[i];
  }
  return left;
}
```

#### [239.滑动窗口最大值](https://leetcode-cn.com/problems/sliding-window-maximum/)

```c_cpp
class Solution {
  using pii = pair<int, int>;
 public:
  vector<int> maxSlidingWindow(vector<int> &nums, int k) {
    //窗口, first = value , second = index
    priority_queue<pii, vector<pii>, less<>> pq;
    for (int i = 0; i < k; i++) {
      pq.emplace(nums[i], i);
    }
    int n = nums.size();
    vector<int> ret;
    ret.push_back(pq.top().first);
    for (int i = k; i < n; i++) {
      pq.emplace(nums[i], i);//窗口插入新的的值
      while (pq.top().second <= i - k)//当前最大值不在窗口中，则弹出，直到当前最大值处于窗口
        pq.pop();
      ret.push_back(pq.top().first);
    }
    return ret;
  }
```

#### [240.搜索二维矩阵 II](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)

```c_cpp
  bool searchMatrix(vector<vector<int>> &matrix, int target) {
    //变种二分查找
    if (matrix.empty())
      return false;
    int m = matrix.size(), n = matrix[0].size();
    int i = 0, j = n - 1;
    while (i < m && j >= 0) {
      if (matrix[i][j] == target)
        return true;
      else if (matrix[i][j] < target) {
        ++i;
      } else {
        --j;
      }
    }
    return false;
  }
```

#### [279.完全平方数](https://leetcode-cn.com/problems/perfect-squares/)

```c_cpp
int numSquares(int n) {
  vector<int> dp(n + 1, 0);
  for (int i = 1; i <= n; i++) {
    dp[i] = i;//最坏情况需要i个1
    for (int j = 1; j * j <= i; j++) {
      //取最小所需的值，i-j*j 表示减去一个完全平方数后所需的数量 + 1(也就是j*j)
      dp[i] = min(dp[i], dp[i - j * j] + 1);
    }
  }
  return dp[n];
}
```

#### [283.移动零](https://leetcode-cn.com/problems/move-zeroes/)

```c_cpp
void moveZeroes(vector<int> &nums) {
  int n = nums.size();
  int i = 0;
  for (int j = 0; j < n; j++) {
    if (nums[j] != 0) {
      swap(nums[i++], nums[j]);
    }
  }
}
```

#### [287.寻找重复数](https://leetcode-cn.com/problems/find-the-duplicate-number/)

```c_cpp
int findDuplicate(vector<int> &nums) {
  //二分查找， 取得mid后， 计算小于等于mid的个数，若<=mid 则在mid右侧，反之亦然；
  int n = nums.size();
  int left = 1, right = nums.size() - 1, ans = -1;
  while (left <= right) {
    int mid = (right - left) / 2 + left;
    int cnt = 0;
    for (int i = 0; i < n; i++) {
      cnt += nums[i] <= mid;
    }
    if (cnt <= mid) {
      left = mid + 1;
    } else {
      right = mid - 1;
      ans = mid;
    }
  }
  return ans;
}
```

#### [297.二叉树的序列化与反序列化](https://leetcode-cn.com/problems/serialize-and-deserialize-binary-tree/)

```c_cpp
class Codec {
 public:

  //先序遍历，空节点使用#代替，使用空格分割
  void serialize(TreeNode *node, string &str) {
    if (node == nullptr) {
      str += "# ";
      return;
    }
    str += to_string(node->val) + ' ';
    serialize(node->left, str);
    serialize(node->right, str);
  }
  // Encodes a tree to a single string.
  string serialize(TreeNode *root) {
    string ret = "";
    serialize(root, ret);
    return ret;
  }

  // Decodes your encoded data to tree.
  TreeNode *deserialize(string data) {
    if (data.empty())
      return nullptr;
    istringstream is(data);
    return deserialize(is);
  }

  TreeNode *deserialize(istringstream &is) {
    string temp;
    is >> temp;
    if (temp == "#")
      return nullptr;
    TreeNode *node = new TreeNode(stoi(temp));
    node->left = deserialize(is);
    node->right = deserialize(is);
    return node;
  }
};
```

#### [300.最长递增子序列](https://leetcode-cn.com/problems/longest-increasing-subsequence/)

```c_cpp
int lengthOfLIS(vector<int> &nums) {
  int n = nums.size();
  //dp[i]表示以 i结尾位置的最长子序长度
  int ans = 1;
  vector<int> dp(n, 1);//每个数字本身就是以自己结尾的长度为1；
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      if (nums[j] < nums[i]) {//若有比前面有比自己小的字符则取最大值 dp[j]+1 包含自己
        dp[i] = max(dp[j] + 1, dp[i]);
      }
    }
    ans = max(dp[i], ans);
  }
  return ans;
}
```

#### [301.删除无效的括号](https://leetcode-cn.com/problems/remove-invalid-parentheses/)

```c_cpp
class Solution {
 public:
  vector<string> res;
  bool isValid(string &str) {
    int cnt = 0;
    for (const auto &item: str) {
      if (item == '(')
        ++cnt;
      else if (item == ')') {
        --cnt;
        if (cnt < 0)
          return false;
      }
    }
    return cnt == 0;
  }

  void backTrack(string str, int start, int lCount, int rCount, int lRemove, int rRemove) {
    if (lRemove == 0 && rRemove == 0) {//需要删除的符号已经删除够，判断是否合法
      if (isValid(str)) {
        res.push_back(str);
      }
      return;
    }

    int size = str.size();
    for (int i = start; i < size; i++) {
      //去重,跳过,跳过前需要记录括号数量
      if (i != start && str[i] == str[i - 1]) {
        //记录左右括号使用的数量
        if (str[i] == '(')
          lCount++;
        else if (str[i] == ')')
          rCount++;
        continue;
      }

      if (lRemove + rRemove > size - i) {//剩余的数量不够删除
        return;
      }

      //删除一个左括号
      if (lRemove > 0 && str[i] == '(')
        backTrack(str.substr(0, i) + str.substr(i + 1), i, lCount, rCount, lRemove - 1, rRemove);

      //删除一个右括号
      if (rRemove > 0 && str[i] == ')')
        backTrack(str.substr(0, i) + str.substr(i + 1), i, lCount, rCount, lRemove, rRemove - 1);

      //记录左右括号使用的数量
      if (str[i] == '(')
        lCount++;
      else if (str[i] == ')')
        rCount++;

      //右括号大于左括号则不合法
      if (rCount > lCount)
        break;
    }

  }

  vector<string> removeInvalidParentheses(string s) {
    int lRemove = 0, rRemove = 0;
    for (const auto &item: s) {
      if (item == '(')
        lRemove++;
      else if (item == ')') {
        if (lRemove == 0)
          rRemove++;
        else
          lRemove--;
      }
    }
    backTrack(s, 0, 0, 0, lRemove, rRemove);
    return res;
  }
};
```

#### [309.最佳买卖股票时机含冷冻期](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/)

```c_cpp
int maxProfit(vector<int> &prices) {
  int n = prices.size();
  /*
   * dp[i][0] 手中持有股票的最大利润
   * dp[i][1] 手中不持有股票,且处于冷冻期的最大利润
   * dp[i][2] 手册不持有股票，且不处于冷冻期的最大利润
   * */
  vector<vector<int>> dp(n, vector<int>(3));
  dp[0][0] = -prices[0];

  for (int i = 1; i < n; ++i) {
    //第i天持有股票最大利润 = max(昨天就持有，昨天不持有不在冷冻期且今天买入）
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i]);
    //第i天不持有股票且在冷冻期（今天要卖掉）, 昨天持有股票+今天卖出价格
    dp[i][1] = dp[i - 1][0] + prices[i];
    //第i天不持有股票，且不在冷冻期(今天不进行任何操作) = max(昨天已经不持有不在冷冻，昨天卖出今天冷冻)
    dp[i][2] = max(dp[i - 1][2], dp[i - 1][1]);
  }
  return max(dp[n - 1][1], dp[n - 1][2]);
}
```

#### [312.戳气球](https://leetcode-cn.com/problems/burst-balloons/)

```c_cpp
int maxCoins(vector<int> &nums) {
  int n = nums.size();
  //修改原始数组，在头尾插入1方便计算
  nums.insert(nums.begin(), 1);
  nums.push_back(1);
  //dp[i][j] 表示(i..j)不包括i和j可以获得的最大硬币数量
  vector<vector<int>> dp(n + 2, vector<int>(n + 2));
  //不断的扩大边界
  for (int i = n - 1; i >= 0; --i) {//左边界
    for (int j = i + 1; j <= n + 1; ++j) {//有边界
      for (int k = i + 1; k < j; ++k) {//k在(i,j)中间，尝试戳破k，取其中的最大值
        int currCoins = nums[i] * nums[k] * nums[j];//错破最后一个气球k获得的硬币数量
        int currTotalCoins = dp[i][k] + currCoins + dp[k][j];//当k为最后一个气球，戳爆后可以获得总硬币数量
        dp[i][j] = max(dp[i][j], currTotalCoins);
      }
    }
  }
  return dp[0][n + 1];
}
```

#### [322.零钱兑换](https://leetcode-cn.com/problems/coin-change/)

```c_cpp
int coinChange(vector<int> &coins, int amount) {
  int Max = amount + 1;//标记，无法兑换
  vector<int> dp(amount + 1, Max);
  std::sort(coins.begin(), coins.end());
  dp[0] = 0;
  for (int i = 1; i <= amount; i++) {
    for (const auto &coin: coins) {
      if (i >= coin)//只有i>=coin时才能够兑换
        dp[i] = min(dp[i], dp[i - coin] + 1);
    }
  }
  return dp[amount] == amount ? -1 : dp[amount];
}
```

#### [338.比特位计数](https://leetcode-cn.com/problems/counting-bits/)

```c_cpp
vector<int> countBits(int n) {
  vector<int> ret(n + 1);
  for (int i = 0; i <= n; i++) {
    ret[i] = count(i);
  }
  return ret;
}
int count(int n) {
  int cnt = 0;
  while (n) {
    n &= (n - 1);//每一次操作消除掉n最后一个1
    ++cnt;
  }
  return cnt;
}
```

#### [347.前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

```c_cpp
//堆排序
vector<int> topKFrequent(vector<int> &nums, int k) {
  priority_queue<pii, vector<pii>, compare> pq;
  unordered_map<int, int> umap;
  for (const auto &item: nums) {
    ++umap[item];
  }
  for (const auto &item: umap) {
    pq.emplace(item.first, item.second);
    if (pq.size() > k) {
      pq.pop();
    }
  }
  vector<int> ans;
  while (!pq.empty()) {
    ans.emplace_back(pq.top().first);
    pq.pop();
  }
  return ans;
}
```

#### [394.字符串解码](https://leetcode-cn.com/problems/decode-string/)

```c_cpp
string decodeString(string s) {
  string ret = "";
  for (const auto &ch: s) {
    if (ch != ']')
      ret.push_back(ch);
    else {
      string str;
      while (!ret.empty() && ret.back() != '[') {
        str += ret.back();
        ret.pop_back();
      }
      ret.pop_back();//弹出[符号
      string num;
      while (!ret.empty() && isdigit(ret.back())) {
        num += ret.back();
        ret.pop_back();
      }
      std::reverse(str.begin(), str.end());
      std::reverse(num.begin(), num.end());
      int n = stoi(num);
      for (int i = 0; i < n; i++) {
        ret += str;
      }
    }
  }
  return ret;
}
```

#### [399.除法求值](https://leetcode-cn.com/problems/evaluate-division/)

```c_cpp
class Solution {
 public:
  int Find(vector<int> &parent, vector<double> &weight, int x) {
    if (parent[x] != x) {
      int father = Find(parent, weight, parent[x]);//递归寻找root节点
      weight[x] *= weight[parent[x]];//更新权值
      parent[x] = father;//设置root节点
    }
    return parent[x];
  }
  void Unite(vector<int> &parent, vector<double> &weight, int x, int y, double value) {
    int x_root = Find(parent, weight, x);//找到x根节点
    int y_root = Find(parent, weight, y);//找到y根节点
    parent[x_root] = y_root;//将x的根节点，指向y_root
    weight[x_root] = value * weight[y] / weight[x];//更新权值
  }
  vector<double> calcEquation(vector<vector<string>> &equations,
                              vector<double> &values,
                              vector<vector<string>> &queries) {
    /*
     * 带权值并查集
     */
    int equation_size = equations.size();
    //将变量与id进行映射绑定
    unordered_map<string, int> variables;
    int id = 0;
    for (int i = 0; i < equation_size; i++) {
      if (variables.find(equations[i].at(0)) == variables.end()) {
        variables[equations[i][0]] = id++;
      }
      if (variables.find(equations[i].at(1)) == variables.end()) {
        variables[equations[i][1]] = id++;
      }
    }
    //初始化每个节点的父节点为本身
    vector<int> parent(id);
    for (int i = 0; i < id; i++) {
      parent[i] = i;
    }
    //初始化权值都为1.0
    vector<double> weight(id, 1.0);

    //合并
    for (int i = 0; i < equation_size; i++) {
      int x = variables[equations[i][0]];
      int y = variables[equations[i][1]];
      Unite(parent, weight, x, y, values[i]);
    }

    //查询
    vector<double> ret;
    for (const auto &q: queries) {
      double result = -1.0;
      if (variables.find(q[0]) != variables.end() && variables.find(q[1]) != variables.end()) {
        int x = variables[q[0]], y = variables[q[1]];
        int fx = Find(parent, weight, x), fy = Find(parent, weight, y);
        if (fx == fy) {//同一个连通分量才能计算
          result = weight[x] / weight[y];
        }
      }
      ret.push_back(result);
    }
    return ret;
  }
};
```

#### [406.根据身高重建队列](https://leetcode-cn.com/problems/queue-reconstruction-by-height/)

```c_cpp
vector<vector<int>> reconstructQueue(vector<vector<int>> &people) {
  /*数组进行排序，按照身高从高到低，若相等，则编号小的在前，这就可以保证每个编号前面的身高都小于或等于自己*/
  std::sort(people.begin(), people.end(), [](const vector<int> &a, const vector<int> &b) {
    return a[0] == b[0] ? a[1] < b[1] : a[0] > b[0];
  });
  /*然后根据编号将对应的人放到合适的位置,频繁修改使用list*/
  list<vector<int>> list;
  for (const auto &p: people) {
    int pos = p[1];
    auto itr = list.begin();
    //找到合适的位置
    while (pos--) {
      itr++;
    }
    //在合适的位置插入
    list.insert(itr, p);
  }
  return vector<vector<int>>(list.begin(), list.end());
}
```

#### [416.分割等和子集](https://leetcode-cn.com/problems/partition-equal-subset-sum/)

```c_cpp
bool canPartition(vector<int> &nums) {
  int n = nums.size();
  //特殊判断
  if (n < 2)
    return false;
  int sum = accumulate(nums.begin(), nums.end(), 0);
  if (sum % 2 != 0)//奇数无法等分
    return false;
  int target = sum / 2;//只要子集能组成target,剩下的也可以
  int maxNum = *max_element(nums.begin(), nums.end());
  if (maxNum > target)//其中一个元素大于target也无法等分
    return false;
  //dp[i][j] 表示 从[0-1]选取一些正整数，每个只能使用一次，使用这些数字的和等于j
  vector<vector<bool>> dp(n, vector<bool>(target + 1, false));
  //第一个数只能使用的容量为第一个数的背包装满
  if (nums[0] <= target) {
    dp[0][nums[0]] = true;
  }
  dp[0][0] = true;
  for (int i = 1; i < n; i++) {
    for (int j = 0; j <= target; j++) {
      dp[i][j] = dp[i - 1][j];
      if (nums[i] <= j) {
        dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i]];
      }
    }
    if (dp[i][target])
      return true;
  }
  return dp[n - 1][target];
}
```

#### [437.路径总和 III](https://leetcode-cn.com/problems/path-sum-iii/)

```c_cpp
//前缀和
unordered_map<int, int> prefix;//key 为前缀和，value为次数
int DFS(TreeNode *node, long long curr, const int target) {
  if (node == nullptr)
    return 0;
  curr += node->val;
  int ret = 0;
  //当前节点的前缀和-target,若存在的话说明从某个节点到当前节点的距离是target
  if (prefix.count(curr - target))
    ret += prefix[curr - target];
  prefix[curr]++;
  ret += DFS(node->left, curr, target);
  ret += DFS(node->right, curr, target);
  prefix[curr]--;
  return ret;
}
int pathSum(TreeNode *root, int targetSum) {
  if (root == nullptr)
    return 0;
  prefix[0] = 1;//初始条件
  return DFS(root, 0, targetSum);
}
// DFS
int target, ans;
//DFS,对应每个节点都进行一次DFS n^2
void DFS(TreeNode *node) {
  DFS(node, 0);
  if (node->left) DFS(node->left);
  if (node->right) DFS(node->right);
}
void DFS(TreeNode *node, int sum) {
  sum += node->val;
  if (sum == target)
    ++ans;
  if (node->left) DFS(node->left, sum);
  if (node->right) DFS(node->right, sum);
}
int pathSum1(TreeNode *root, int targetSum) {
  if (root == nullptr)
    return 0;
  target = targetSum;
  ans = 0;
  DFS(root);
  return ans;
}
```

#### [438.找到字符串中所有字母异位词](https://leetcode-cn.com/problems/find-all-anagrams-in-a-string/)

```c_cpp
vector<int> findAnagrams(string s, string p) {
  //记录p每个字符出现的次数
  vector<int> needs(26);
  for (const auto &item: p) {
    ++needs[item - 'a'];
  }
  vector<int> wid_cnt(26);//滑动窗口的大小
  int left = 0;
  int n = s.size();
  vector<int> ret;
  for (int right = 0; right < n; right++) {
    //当前出现的字符
    int cur_right = s[right] - 'a';
    ++wid_cnt[cur_right];
    while (wid_cnt[cur_right] > needs[cur_right]) {//滑动窗口与needs中相同的字符较多，需要删除
      //需要删除的字符
      int cur_left = s[left] - 'a';
      --wid_cnt[cur_left];//删除
      ++left;//移动左边界
    }
    if (right - left + 1 == p.size())
      ret.push_back(left);
  }
  return ret;
}
```

#### [448.找到所有数组中消失的数字](https://leetcode-cn.com/problems/find-all-numbers-disappeared-in-an-array/)

```c_cpp
vector<int> findDisappearedNumbers(vector<int> &nums) {
  //标记法
  int n = nums.size();
  for (const auto &item: nums) {//每个数字对应的下标 为 item-1;
    if (nums[abs(item) - 1] > 0) { //将出现数字的下标标记位负数,没出先过的数字对应的下标就会为正数
      nums[abs(item) - 1] *= -1;
    }
  }
  vector<int> ret;
  for (int i = 0; i < n; i++) {
    if (nums[i] > 0) {
      //注意下标对应的数字 下标+1
      ret.push_back(i + 1);
    }
  }
  return ret;
}
//暴力
vector<int> findDisappearedNumbers1(vector<int> &nums) {
  int n = nums.size();
  //记录每个数字出现的次数，并记录到对应的位置
  vector<int> temp(n + 1);
  for (const auto &item: nums) {
    ++temp[item];
  }
  vector<int> ret;
  //若i位置出现为0则缺少该数字
  for (int i = 1; i <= n; i++) {
    if (temp[i] == 0)
      ret.push_back(i);
  }
  return ret;
}
```

#### [461.汉明距离](https://leetcode-cn.com/problems/hamming-distance/)

```c_cpp
int hammingDistance(int x, int y) {
  int cnt = 0;
  int xorNum = x ^ y;//异或之后，不同的结果为1,1的个数即为不同的个数
  while (xorNum) {
    if (xorNum & 1) {
      ++cnt;
    }
    xorNum >>= 1;//右移一位
  }
  return cnt;
}
```

#### [494.目标和](https://leetcode-cn.com/problems/target-sum/)

```c_cpp
//动态规划
int findTargetSumWays(vector<int> &nums, int target) {
  /*
   *设所有元素的和sum, 添加-号的元素和为neg 则有
   * (sum-neg) - neg = target  => sum-2 neg = target => neg =( sum-target) /2
   * 问题转为为01背包问题，从num中选取一些元素使得其和为neg
   * */
  int sum = accumulate(nums.begin(), nums.end(), 0);
  int diff = sum - target;
  if (diff < 0 || diff % 2 != 0)
    return 0;
  int n = nums.size(), neg = diff / 2;
  vector<vector<int>> dp(n + 1, vector<int>(neg + 1));
  dp[0][0] = 1;
  for (int i = 1; i <= n; i++) {
    int num = nums[i - 1];//当前数字
    for (int j = 0; j <= neg; j++) {
      dp[i][j] = dp[i - 1][j];//若不选择
      if (j >= num) {//j>=num可以选择
        dp[i][j] += dp[i - 1][j - num];
      }
    }
  }
  return dp[n][neg];
}
//回溯
int ret = 0;
int findTargetSumWays1(vector<int> &nums, int target) {
  backTrack(nums, 0, 0, target);
  return ret;
}
void backTrack(const vector<int> &nums, int sum, int index, int target) {
  if (index == nums.size()) {
    if (sum == target)
      ret++;
    return;
  }

  backTrack(nums, sum + nums[index], index + 1, target);
  backTrack(nums, sum - nums[index], index + 1, target);
}
```

#### [538把二叉搜索树转换为累加树](https://leetcode-cn.com/problems/convert-bst-to-greater-tree/)

```c_cpp
TreeNode *convertBST1(TreeNode *root) {
  int value = 0;
  DFS(root, value);
  return root;
}
//反中序遍历
void DFS(TreeNode *node, int &value) {
  if (node == nullptr)
    return;
  DFS(node->right, value);
  node->val += value;
  value = node->val;
  DFS(node->left, value);
}
```

#### [543.二叉树的直径 ](https://leetcode-cn.com/problems/diameter-of-binary-tree/)

```c_cpp
int ans = 0;
int diameterOfBinaryTree(TreeNode *root) {
  maxDeep(root);
  return ans;
}
int maxDeep(TreeNode *node) {
  if (node == nullptr)
    return 0;
  int left = maxDeep(node->left);
  int right = maxDeep(node->right);
  ans = max(ans, left + right);//任意两点的距离的最长直接为某个节点的左右子树深度的和
  return max(left, right) + 1;
}
```

#### [560.和为 K 的子数组](https://leetcode-cn.com/problems/subarray-sum-equals-k/)

```c_cpp
int subarraySum(vector<int> &nums, int k) {
  unordered_map<int, int> hash;//key为钱缀和，value为出现的次数
  hash[0] = 1;//考虑前缀恰好等于k
  int cnt = 0;
  int pre = 0;
  for (const auto &n: nums) {
    pre += n;
    if (hash.count(pre - k))
      cnt += hash[pre - k];
    ++hash[pre];
  }
  return cnt;
}
```

#### [581.最短无序连续子数组](https://leetcode-cn.com/problems/shortest-unsorted-continuous-subarray/)

```c_cpp
  int findUnsortedSubarray(vector<int> &nums) {
    if (is_sorted(nums.begin(), nums.end()))
      return 0;

    vector<int> temp(nums);
    std::sort(temp.begin(), temp.end());
    int left = 0;
    while (nums[left] == temp[left])
      ++left;
    int right = nums.size() - 1;
    while (nums[right] == temp[right])
      --right;
    return right - left + 1;
  }
```

#### [617.合并二叉树](https://leetcode-cn.com/problems/merge-two-binary-trees/)

```c_cpp
TreeNode *mergeTrees(TreeNode *root1, TreeNode *root2) {
  if (root1 == nullptr)
    return root2;
  if (root2 == nullptr)
    return root1;
  auto newRoot = new TreeNode(root1->val + root2->val);
  newRoot->left = mergeTrees(root1->left, root2->left);
  newRoot->right = mergeTrees(root1->right, root2->right);
  return newRoot;
}
```

#### [621.任务调度器](https://leetcode-cn.com/problems/task-scheduler/)

```c_cpp
int leastInterval(vector<char> &tasks, int n) {
  int size = tasks.size();
  if (n == 0)
    return size;
  //统计任务数量
  vector<int> alpha(26, 0);
  for (const auto &item: tasks) {
    ++alpha[item - 'A'];
  }
  int maxFreq = 0, maxTasks = 0;//最大任务的次数，最大次数任务的数量
  for (const auto &item: alpha) {
    if (item > maxFreq) {
      maxFreq = item;
      maxTasks = 1;
    } else if (maxFreq == item) {
      ++maxTasks;
    }
  }
  int ret = (maxFreq - 1) * (n + 1) + maxTasks;
  return max(size, ret);
}
```

#### [647.回文子串](https://leetcode-cn.com/problems/palindromic-substrings/)

```c_cpp
int countSubstrings(string s) {
  int n = s.size();
  vector<vector<bool>> dp(n, vector<bool>(n, false));
  for (int i = 0; i < n; i++) {
    dp[i][i] = true;
  }
  int ret = n;//默认每单个字符都是
  for (int right = 0; right < n; right++) {
    for (int left = 0; left < right; left++) {
      if (s[left] != s[right])
        continue;
      //当left与right相等时且<3直接为true, 若>=3 则看里层是否是回文串
      dp[left][right] = right - left < 3 || dp[left + 1][right - 1];
      if (dp[left][right])
        ret++;
    }
  }
  return ret;
}
```

#### [739.每日温度](https://leetcode-cn.com/problems/daily-temperatures/)

```c_cpp
vector<int> dailyTemperatures(vector<int> &temperatures) {
  int n = temperatures.size();
  vector<int> ret(n, 0);
  stack<int> st;
  for (int i = 0; i < n; i++) {
    //保持单调递减,遇到更高的温度，则计算
    while (!st.empty() && temperatures[st.top()] < temperatures[i]) {
      ret[st.top()] = i - st.top();
      st.pop();
    }
    st.push(i);
  }
  return ret;
}
vector<int> dailyTemperatures1(vector<int> &temperatures) {
  //暴力
  int n = temperatures.size();
  vector<int> ret(n);
  for (int i = 0; i < n; i++) {
    int cnt = 0;
    for (int j = i + 1; j < n; j++) {
      ++cnt;
      if (temperatures[j] > temperatures[i]) {
        ret[i] = cnt;
        break;
      }
    }
  }
  return ret;
}
```

<br/>

<br/>

<br/>

<br/>

<br/>

#### [207. 课程表](https://leetcode-cn.com/problems/course-schedule/)

```c_cpp
/*bfs*/
bool canFinish(int numCourses, vector<vector<int>> &prerequisites) {
    vector<int> inDegree(numCourses);/*记录课程的入读*/
    unordered_map<int, vector<int>> edges;/*记录前置课程和与之相关的课程*/

    for (const auto &course : prerequisites) {
        inDegree[course[0]]++;
        edges[course[1]].emplace_back(course[0]);
    }

    queue<int> queue;
    /*将入度为0的先入对(即没有前置课程的先入队)*/
    for (int i = 0; i < inDegree.size(); ++i) {
        if (inDegree[i] == 0) {
            queue.push(i);
        }
    }
    int finishCount = 0;
    while (!queue.empty()) {
        auto course = queue.front();
        queue.pop();
        for (const auto &relationCourse : edges[course]) {/*相关课程的入度减少，若有课程度数为0则入对*/
            inDegree[relationCourse]--;
            if (inDegree[relationCourse] == 0) {
                queue.push(relationCourse);
            }
        }
        finishCount++;
    }
    return finishCount == numCourses;
}
```

#### [4.寻找两个正序数组的中位数](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)

```c_cpp
double findMedianSortedArrays(vector<int> &nums1, vector<int> &nums2) {
  //二分法
  int n = nums1.size(), m = nums2.size();
  //保证第一个数组为较小的那个
  if (m < n)
    return findMedianSortedArrays(nums2, nums1);

  int LMax1 = 0, RMin1 = 0, LMax2 = 0, RMin2 = 0;
  int lo = 0, hi = n;//二分范围，以最小的数组为
  while (lo <= hi) {
    int cut1 = (hi - lo + 1) / 2 + lo;//第一个数组切割的位置
    int cut2 = (m + n) / 2 - cut1;//第二数组切割的位置= 需要寻找中位数的位置-已经在数组一切割的数量

    LMax1 = cut1 == 0 ? INT_MIN : nums1[cut1 - 1];//第一个数组若切割到0的位置，则左侧没有数据，使用INT_MIN表示，确保足够小
    RMin1 = cut1 == n ? INT_MAX : nums1[cut1];//第一个数组若切割到n的位置，则右侧没有数据，使用INT_MAX表示，确保足够大
    LMax2 = cut2 == 0 ? INT_MIN : nums2[cut2 - 1];
    RMin2 = cut2 == m ? INT_MAX : nums2[cut2];

    /* 说明C2位置过小，需要扩大，C2由C1决定，而C1的大小取决于【lo,hi】,因此缩小hi*/
    if (LMax1 > RMin2)
      hi = cut1 - 1;
      /* 说明C1位置过小，需要扩大C1,而C1的大小取决于【lo,hi】,因此扩大lo*/
    else if (LMax2 > RMin1)
      lo = cut1 + 1;
    else/*若不满足上述情况，则找到了两个合适的分割点，退出即可*/
      break;
  }
  if ((m + n) % 2 != 0)//奇数直返最小值
    return min(RMin1, RMin2);
  //偶数情况，左边取最大，右边取最小
  return (max(LMax1, LMax2) + min(RMin1, RMin2)) / 2.0;
}

double findMedianSortedArrays1(vector<int> &nums1, vector<int> &nums2) {
  //合并两个数组，然后找到中位数
  int m = nums1.size(), n = nums2.size();
  //特殊情况判断
  if (m == 0) {
    return n % 2 == 0 ? (double) (nums2[n / 2] + nums2[n / 2 - 1]) / 2 : nums2[n / 2];
  }
  if (n == 0) {
    return m % 2 == 0 ? (double) (nums1[m / 2] + nums1[m / 2 - 1]) / 2 : nums1[m / 2];
  }
  //合并数组
  int size = m + n;
  vector<int> nums(size);
  int i = 0, j = 0, k = 0;
  while (k < size) {
    if (nums1[i] < nums2[j]) {
      nums[k++] = nums1[i++];
    } else {
      nums[k++] = nums2[j++];
    }

    if (i == m) {
      while (j < n) {
        nums[k++] = nums2[j++];
      }
    }
    if (j == n) {
      while (i < m) {
        nums[k++] = nums1[i++];
      }
    }
  }
  return size % 2 == 0 ? (double) (nums[size / 2] + nums[size / 2 - 1]) / 2 : nums[size / 2];
}
```

<br/>

<br/>

##  五、每日一题

#### [372. 超级次方](https://leetcode-cn.com/problems/super-pow/)

```c_cpp
/*前提：快速幂，重点拆分数组时的次幂计算*/
const int mod = 1337;
int superPow(int a, vector<int> &b) {
    int res = 1;
    /*从后面往前遍历*/
    /*eg a=2 b=[1,2,3]*/
    /* (2^[1,2])^10 * 2^3 */
    for (int i = b.size() - 1; i >= 0; --i) {
        res = (long )res* myPow(a,b[i])%mod;
        a = myPow(a,10);
    }
    return res;
}
/*快速幂*/
int myPow(int x, int y) {
    int res = 1;
    while (y) {
        if ((y & 1) == 1)
            res = (long) res * x % mod;
        x = (long) x * x % mod;
        y >>= 1;
    }
    return res;
}
```

#### [1034. 边界着色](https://leetcode-cn.com/problems/coloring-a-border/)

```c_cpp
/*重点如何判断当前是否是边界节点*/
vector<vector<int>> colorBorder(vector<vector<int>> &grid, int row, int col, int color) {
    int m = grid.size();
    int n = grid[0].size();
    vector<vector<bool>> visited(m, vector<bool>(n));/*用于标记是否使用*/
    visited[row][col] = true;/*第一个直接为true*/
    vector<pair<int, int>> border;/*需要上色的边界*/
    int originalColor = grid[row][col];
    DFS(grid, visited, border, row, col, originalColor);
    for (const auto&[x, y] : border) {
        grid[x][y] = color;
    }
    return grid;
}
void DFS(vector<vector<int>> &grid,
         vector<vector<bool>> &visited,
         vector<pair<int, int>> &border,
         int x,
         int y,
         int originalColor) {
    int m = grid.size();
    int n = grid[0].size();
    /*四个方向*/
    int direct[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
    for (int i = 0; i < 4; ++i) {
        /*获得下一个方向的坐标*/
        int nx = direct[i][0] + x, ny = direct[i][1] + y;
        /*下一个位置超出边界或颜色不等，说明当前xy为连通分量的边界点*/
        if (!(nx < m && nx >= 0 && ny < n && ny >= 0 && grid[nx][ny] == originalColor)) {
            border.push_back({x, y});
        } else if (!visited[nx][ny] && grid[nx][ny] == originalColor) {/*下一个位置没有访问过，且颜色相等属于同一个连通分量,则递归*/
            visited[nx][ny] = true;
            DFS(grid, visited, border, nx, ny, originalColor);
        }
    }
}
```

#### [851. 喧闹和富有](https://leetcode-cn.com/problems/loud-and-rich/)

```c_cpp
vector<int> loudAndRich(vector<vector<int>> &richer, vector<int> &quiet) {
    /*将其抽象成一张有向图，每个人指向比自己少钱的人*/
    int size = quiet.size();
    vector<int> inDegree(size);/*记录每个人的入度*/
    vector<vector<int>> edges(size);/*key为节点，value为指向比自己少钱的人*/
    /*构建图和计算每个节点的入度*/
    for (const auto &rich : richer) {
        inDegree[rich[1]]++;
        edges[rich[0]].push_back(rich[1]);
    }
    vector<int> ans(size);
    /*初始化ans，默认每个人自己最安静的人*/
    iota(ans.begin(), ans.end(), 0);
    queue<int> queue;
    /*入度为0的先入度*/
    for (int i = 0; i < size; ++i) {
        if (inDegree[i] == 0)
            queue.push(i);
    }
    while (!queue.empty()) {
        auto x = queue.front();
        queue.pop();

        for (const auto &y : edges[x]) {
            if (quiet[ans[x]] < quiet[ans[y]])/*富有节点的安静值小于平穷节点的安静值，则更新平穷节点的答案*/
                ans[y] = ans[x];
            if (--inDegree[y] == 0)/*入度为0的节点入度*/
                queue.push(y);
        }
    }
    return ans;
}
```

#### [1705. 吃苹果的最大数目](https://leetcode-cn.com/problems/maximum-number-of-eaten-apples/)

```c_cpp
/*贪心+优先队列*/
int eatenApples(vector<int> &apples, vector<int> &days) {
  /*二元数组，first为腐烂日期，second为苹果数量*/
  priority_queue<pair<int, int>, vector<pair<int, int>>, greater<>> pq;
  int res = 0;
  int i = 0;
  int n = apples.size();
  while (i < n) {
    if (!pq.empty() && pq.top().first <= i) {/*将腐烂的出队*/
      pq.pop();
    }
    /*计算腐烂日期和当天产量，如果>0则入队*/
    int rottenDay = i + days[i];
    int count = apples[i];
    if (count > 0) {
      pq.emplace(rottenDay, count);
    }
    /*吃苹果*/
    if (!pq.empty()) {
      auto apple = pq.top();
      pq.pop();
      apple.second--;
      if (apple.second != 0)/*当前这批水果还没吃完，入队*/
        pq.emplace(apple.first, apple.second);
      res++;
    }
    i++;
  }
  /*处理剩下的苹果*/
  while (!pq.empty()) {
    while (!pq.empty() && pq.top().first <= i) {/*将过期苹果出队*/
      pq.pop();
    }
    if (pq.empty())
      break;
    auto apple = pq.top();
    pq.pop();
    int num = min(apple.first - i, apple.second);/*判断当前这批水果还能吃多少天*/
    res += num;
    i += num;
  }
  return res;
}
```

#### [540.有序数组中的单一元素](https://leetcode-cn.com/problems/single-element-in-a-sorted-array/)

- 特殊技巧：**连续段 偶奇 两两成对**，可以使用**异或**寻找组成对象。 如2^1 =3   3^1=2;

```c_cpp
int singleNonDuplicate(vector<int> &nums) {
  /*
   * 二分法：思路，由于数组从0开始，因此若出现成对相等的元素，则 第一个数为偶数位，第二位为奇数为，利用这个行政左二分
   */
  int n = nums.size();
  int left = 0, right = n - 1;
  while (left < right) {
    int mid = (right - left) / 2 + left;
    //特殊技巧：连续段 偶奇 两两成对，可以使用异或寻找组成对象。 如2^1 =3   3^1=2;
    if (nums[mid] == nums[mid ^ 1])
      left = mid + 1;
    else
      right = mid;
  }
  return nums[right];
}
```
