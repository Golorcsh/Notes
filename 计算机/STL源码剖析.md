## 侯捷——STL源码剖析 笔记

## 1.总览

### 1.STL六大部件之间的关系

![在这里插入图片描述](https://img-blog.csdnimg.cn/6094727e90ae4f398abcb5a3286ef463.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

在下图中，我们使用了如下：  
1.一个**容器**[vector](https://so.csdn.net/so/search?q=vector&spm=1001.2101.3001.7020)  
2.使用vector时，使用**分配器**分配[内存](https://so.csdn.net/so/search?q=%E5%86%85%E5%AD%98&spm=1001.2101.3001.7020)  
3.使用vi.begin(),vi.end()即**迭代器**，作为算法的参数  
4.使用count\_if**算法**  
5.使用**仿函数**less()  
6.使用函数**适配器**来对我们算法的结果进行进一步筛选（not1, bind2nd）  
![在这里插入图片描述](https://img-blog.csdnimg.cn/d4e0e413d6c7455d9195530903d24b36.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.复杂度

![在这里插入图片描述](https://img-blog.csdnimg.cn/756b30ceddd1452987ae2d4259d2ca2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.[迭代器](https://so.csdn.net/so/search?q=%E8%BF%AD%E4%BB%A3%E5%99%A8&spm=1001.2101.3001.7020)区间

迭代器是一个左开右闭的区间，也就是说迭代器的end是最后一个元素的下一个元素。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/6c6e9abdbcde47bdbb902d9db1ae8cf3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 4.容器的结构和分类

![在这里插入图片描述](https://img-blog.csdnimg.cn/91745a9bbaa4457d926807935b79d80a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 2.容器分类及测试

### 1.序列式容器

##### 序列式容器的特点是将数据放进容器之后，会按照用户放进去的顺序依次排列

|序列式容器|特点|额外学习材料|
|--|--|--|
|array|一段连续空间，不论是否使用，都会全部占用|[array](http://m.biancheng.net/view/6688.html)|
|vector|尾部可进可出，当空间不够时会自动扩充|[vector](http://m.biancheng.net/view/6749.html)|
|deque|双向都可扩充，两端都可进可出|[deque](http://m.biancheng.net/view/6860.html)|
|list|一个双向环状链表，有向前后和向后两个指针|[list](http://m.biancheng.net/view/6892.html)|
|forward\_list|一个单向链表，仅有向后一个指针|[forward\_list](http://m.biancheng.net/view/6960.html)|

![在这里插入图片描述](https://img-blog.csdnimg.cn/7b8e0c0060ed436cad57a83fa2877e1c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.关联式容器

##### 关联式容器类似于key-value，非常适合于查找操作

|关联式容器名|特点|实现|注释|额外学习材料|
|--|--|--|--|--|
|set/multiset|key和value是同一个，BST存储是有序的|红黑树|加上multi意味着可以重复键值对|[set](http://m.biancheng.net/view/7192.html)，[multiset](http://m.biancheng.net/view/7203.html)|
|map/multimap|每一个key对应一个value，BST存储是有序的|红黑树|加上multi意味着可以重复键值对|[map](http://m.biancheng.net/view/7173.html)，[multimap](http://m.biancheng.net/view/7190.html)|
|unordered\_set/unordered\_multiset|相对于set/multiset，存储是无序的|哈希表|加上multi意味着可以重复键值对|[unordered\_set](http://m.biancheng.net/view/7250.html)，[unordered\_multiset](http://m.biancheng.net/view/7251.html)|
|unordered\_map/unordered\_multimap|相对于map/multimap，存储是无序的|哈希表|加上multi意味着可以重复键值对|[unordered\_map](http://m.biancheng.net/view/7231.html)，[unordered\_multimap](http://m.biancheng.net/view/7248.html)|

![在这里插入图片描述](https://img-blog.csdnimg.cn/f3f8bd5d6bce4ad58084e92a717ef69d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 3.分配器(Allocator)详解

### 1.总览

分配器的效率非常重要。因为容器必然会使用到分配器来负责内存的分配，它的性能至关重要。

##### 在C++中，内存分配和操作通过new和delete完成。

new中包含两个操作，第一步是使用operator new分配内存，第二步是调用构造函数；  
delete中包含两个操作，第一步是调用析构函数，第二步是使用operator delete释放内存。

#### 1.分配器底层都会回到malloc

C++的内存分配动作最终都会回到malloc，malloc再根据不同的操作系统类型(Windows，Linux，Unix等)底层的系统API来获取内存

##### 同时我们可以看到，malloc分配之后的内存块中不是只有数据，而是还包含了其它很多数据。这样容易联想到如果分配次数越多，那么内存中数据越零散，这些额外的数据开销就越大。

##### 所以一个优秀的分配器，应当尽可能的让这些额外的空间占比更小，让速度更快。

![在这里插入图片描述](https://img-blog.csdnimg.cn/1b6da608b7e345d59da185a583cafb84.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.VC6，BC5，GC2.9所带的标准库分配器源码分析

上面我们提到了分配器的评判标准，现在我们来看一下编译器自带的标准库中的分配器是如何实现的。  
VC6，BC5，GC2.9的标准库分配器并没有做特殊设计。就是调用malloc和free。  
缺点如下：

1.接口设计不方便。如果我们单独调用分配器，那么我们需要记住我们指向分配的那片内存空间的指针，以及分配的内存空间大小。不然我们无法使用deallocate来释放这份空间。虽然容器不会有影响。

2.如果我们需要多次分配空间，默认的分配器由于每次分配的空间都很小，导致我们需要进行很多次内存分配的操作，同时需要很多额外空间。那么这个没有特殊设计过的分配器在这种情况下的效率就会变得低下，影响程序运行效率  
![在这里插入图片描述](https://img-blog.csdnimg.cn/22281e5dcffe4c4387b46fb1093a9206.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

BC5的分配器与VC6没有本质区别。BC5的优点是他的分配器第二参数有一个默认值，让我们在调用分配器时方便了一些。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/8aef58a9e9e74cb3bf6dd0ba0ca84268.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

GC2.9自带的allocator也差不多  
![在这里插入图片描述](https://img-blog.csdnimg.cn/31f2bd67cbef498789c2b430ef3cbdc5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### 虽然GC2.9和上面也基本一致，但是它有额外声明不要使用这个标准库的分配器，同时这个标准库分配器没有被使用。它使用的分配器是自行修改的

GC2.9使用的是一个叫**alloc**的分配器  
![在这里插入图片描述](https://img-blog.csdnimg.cn/1c246a97e1814191aff2c31f8b942abf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.GC2.9的分配器的效率提高思路

#### 1.内存空间简介

通过面向对象高级编程(上)的学习，我们可以将malloc分配出的内存区块分为这几个部分。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/4f3976df5b6644ab8816502dd3eb4a18.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.G2.9 分配器——alloc

从上面对内存空间的分析可以知道，malloc分配出的内存区块中需要有地方来存放这个内存区块的大小。然而对于同一个容器而言，它的内置类型应当是相同的，所以对于容器的分配器，我们可以对此作出优化。  
alloc创建了16条单向链表用来存放数据。这些单向链表用来存放不同元素大小的数据。  
当容器需要内存时，alloc先查看自己是否已经申请过了这个大小的内存，如果已经申请过了，那么就继续放在对应的单向链表尾部。否则再调用malloc向系统申请一块内存空间。具体可以查看这里[【C++内存管理】G2.9 std::alloc 运行模式](https://blog.csdn.net/ZLP_CSDN/article/details/106427351)  
它的优点就是，由于每个链表都只有一种大小的元素，那么对于这条链表上的每一个元素，我们就不必再单独使用内存空间来记录它的大小。从而节省了内存空间  
![在这里插入图片描述](https://img-blog.csdnimg.cn/b4cd0083c8d54ceb8f897077affee289.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)  
在这张图中可以看到很多看起来非常杂乱的连线，这个实际上是alloc的内存申请机制影响的，alloc在申请内存时会考虑之前剩余下来的内存余量（这里存在pool当中），如果有内存余量的话在下一次申请空间时，会将上一次分配剩下来的内存空间按照需要的大小进行切割并挂载到对应的节点上。如果上一次剩余的大小不足以划分，那么会将这个剩余的内存空间挂到与它相等的内存空间大小的节点上去，然后重新分配内存。具体可以参考[这里](https://blog.csdn.net/qq_34269632/article/details/115636008)  
在G4.9中，分配器变成了new\_allocator，旧的分配器alloc改名为\_pool\_alloc。

### 4.(补充)SGI的两级分配器

![在这里插入图片描述](https://img-blog.csdnimg.cn/7e187500ed164085a9c9b694c7f05875.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

STL的分配器用于封装STL容器在内存管理上的底层细节。在C++中，其内存配置和释放如下：

##### new运算分两个阶段：

(1)调用::operator new配置内存;  
(2)调用对象构造函数构造对象内容

##### delete运算分两个阶段：

(1)调用对象析构函数；  
(2)调用::operator delete释放内存  
为了精密分工，STL allocator将两个阶段操作区分开来：  
内存配置有alloc::allocate()负责，内存释放由alloc::deallocate()负责；  
对象构造由::construct()负责，对象析构由::destroy()负责。  
同时为了提升内存管理的效率，减少申请小内存造成的内存碎片问题，SGI STL采用了两级配置器  
当分配的空间大小超过128B时，会使用第一级空间配置器；当分配的空间大小小于128B时，将使用第二级空间配置器。  
第一级空间配置器直接使用malloc()、realloc()、free()函数进行内存空间的分配和释放，而第二级空间配置器采用了内存池技术，通过空闲链表来管理内存。  
当然，alloc也可以直接作为第一级分配器。

## 4.深度探索list

### 1.list的基本组成

list是一个双向链表，它的基本组成就是

|成员|作用|
|--|--|
|prev指针|指向上一个元素|
|next指针|指向下一个元素|
|data|用来保存数据|

![在这里插入图片描述](https://img-blog.csdnimg.cn/2f101f25e9a2456386ee08aad0c7fce9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.list的迭代器

由于人们一般习惯于：迭代器++是找到下一个元素，迭代器–是找到上一个元素。在双向链表list中，我们可以知道下一个元素就是next所指元素，上一个元素就是prev所指元素。  
如果我们想要实现迭代器++的操作，就需要访问list节点对应的next指针。所以迭代器是一个类，需要为我们封装这些操作，或者更准确的说，迭代器类是一个**智能指针**。

##### list的插入和接合操作都不会造成原有的list迭代器失效，对于删除操作，也只有”指向被删除元素“的那个迭代器失效，其它迭代器不受任何影响

#### 1.++i 和 i++的重载

##### Q：在C++中，由于++i和i++都只有一个参数，那么如何对这两种分别进行重载呢？

##### A：在C++中，规定了带有参数的是后置++，没有参数的是前置++。比如说

```c_cpp
operator++(int) {}; //对 i++ 进行重载
operator++() {};    //对 ++i 进行重载
```

#### 2.注意点：

1.后置++的\* 操作符不是解引用，而是调用了拷贝构造函数来制造一个副本  
2.为了模拟C++的整数不能进行如下操作：

```c_cpp
(i++)++;    //不允许
i++++;     //不允许
(++i)++;    //允许
++++i;      //允许
```

C++允许前置++连续，但是不允许后置++连续，所以迭代器中，对于前置++，返回的是引用。而后置++运算符返回的不是reference，而是值；  
![在这里插入图片描述](https://img-blog.csdnimg.cn/5bb4864788b54d67a455dfed09e3ff50.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/8dc770364a3e4a99a5c38f2da7b4334f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.G4.9的list

#### 1.G4.9对比G2.9的一些细节修正

##### 1.list中指针的类型不再是void\*

##### 2.代器不再需要传一种类型的三个形式（T,\* T,& T），而是传入T之后再typedef。

![在这里插入图片描述](https://img-blog.csdnimg.cn/78196a8e67194290bc4c6ee90677fe38.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.G4.9的list更加复杂

![在这里插入图片描述](https://img-blog.csdnimg.cn/d14fd7e26faa4e848afa58d75982fce3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 5.迭代器补充

### 1.迭代器的设计原则

迭代器是算法和容器之间的桥梁，所以算法会想知道迭代器的一些性质来辅助算法。  
这些性质如下：

|五种迭代器中必须typedef的性质|解释|
|--|--|
|iteratior\_category|迭代器类型|
|value\_type|迭代器所指对象的类型|
|difference\_type|两个相邻的迭代器之间的距离|
|pointer|指向value type的指针|
|reference|对value type的引用|

![在这里插入图片描述](https://img-blog.csdnimg.cn/45ff92e2421e49b2b2443e1bb75fce95.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.iterator traits的作用和设计

#### 1.作用

由于上面的设计原则可以知道，迭代器必须typedef五个性质。但是如果这个指针不是一个class的指针，而就是一个普通的指针的话，这样的话，我们怎么分辨呢？iterator traits就用上了。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/fe7f7500f3884137ac90ceec4d2ab8af.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.设计

设计一个中间层作在迭代器和算法中间作为媒介，这个中间层就是iterator traits

##### 实际上就是利用了C++中模板的偏特化来进行一个区分。

##### 注意即使是const指针，为了它能够创建一个非const变量，我们也应当返回一个非const的类型。

图1这里仅仅是举例，完整在图2  
![在这里插入图片描述](https://img-blog.csdnimg.cn/80195a3479d648ba88fcd33d412341b1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/d8b8b29811c9484c9bf4938ebe28ed9a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 6.深度探索vector

vector就是一个可以自动扩充的array。

### 1.源码解析

vector主要是通过三个指针来维护的，分别是起点，当前终点，以及当前最大空间  
![在这里插入图片描述](https://img-blog.csdnimg.cn/baee9b15fc2848919b43e5b1f06f2ddf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.vector的增长形式——两倍增长

vector每当遇到空间不同的情况，都会按照当前最大空间的两倍空间进行空间申请。vector每次扩张都会视本身元素个数多少而造成元素的拷贝，以及元素的删除。  
如果申请不到两倍大的空间，生命就会自动结束。  
[面试题：C++vector的动态扩容，为何是1.5倍或者是2倍？](https://blog.csdn.net/qq_44918090/article/details/120583540)

#### 1.自制的vector增长流程图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c2d5345254d24993827d507695c8c5a2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.代码

![在这里插入图片描述](https://img-blog.csdnimg.cn/3c60201f3f944d0e966ae090df3f4c49.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/ce133b22eba94bdca63279e3df5de52a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.vector的迭代器

##### 注意：插入操作可能造成vector的3个指针重新配置，导致原有的迭代器全部失效

#### 1.G2.9版本的vector迭代器

由于vector本身就是连续的，内存也是连续的，所以正常来讲vector的迭代器不必设置的非常复杂，只需要一个指针就够了。事实上，G2.9中确实是这么做的。  
在G2.9版本中，vector的迭代器就是一个指针。如果将它放入iterator traits当中的话，由于这个迭代器是单独的指针而不是一个类，所以会走偏特化的路线来为算法提供所需的性质。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/ca3484f922f3490cae9264334a0a32f2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.G4.9版本的vector迭代器

然而在G4.9中，vector的迭代器被设计的十分复杂，同时变成了一个类。所以G4.9之后的vector迭代器不会再走指针偏特化的iterator traits了。  
但是这个操作十分的复杂，而且并没有影响最终的结果，也就是说**G2.9和G4.9的迭代器并没有什么本质区别**。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/9ec69963415a4c05985cb6006f2641d0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 7.深度探索array与forward\_list

array就是一个固定长度的数组。

##### Q：为什么要将array设计成一个容器类呢？

##### A：因为这样可以让array返回算法需要的五个性质，这样算法可以适配array来进行一些操作来提高算法性能。

### 1.array源码解析

在TR1中，array的源代码比较简洁。  
由于是连续空间，所以迭代器就用指针实现，通过iterator traits时走偏特化路线。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/0747383044ad40d98700aba61289e5f2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

在G4.9中，array的源代码和上面的vector一样变得复杂，它的迭代器变成了类  
实际上和TR1中最终效果并没有区别。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/88b53f902742491c827cbb146b6d5773.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.forward\_list源码解析

forward\_list就是前面的list少一个向前的指针，参考前面的list即可  
![在这里插入图片描述](https://img-blog.csdnimg.cn/c7e67fef4b84404f943dca7642f787cf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 8.深度探索deque，queue，stack

### 1.deque

#### 1.deque结构总览

deque对外是连续的，但内部不是连续的。

|deque组件|作用|
|--|--|
|一个vector|从来存放那些内存空间的迭代器位置，模仿“连续”|
|若干buffer大小的内存空间|用来保存数据|

deque的迭代器中的四个指针的作用分析

|名称|作用|
|--|--|
|cur|当前buffer上当前节点的位置|
|first|当前buffer上头部节点的位置|
|last|当前buffer上尾部节点的位置|
|node|当前buffer在map上的位置|

当每一个buffer大小的内存空间不够用时，vector会在尾部创建一个迭代器指针并申请一个buffer大小的新空间来放元素。如果vector本身不够大，那么vector会自行扩容。扩容之后需要进行元素的重新拷贝，由于deque是一个**双向**队列，它会将原来的元素拷贝到扩充完了的vector的**中段**，这样的deque就同时拥有向左和向右扩张的能力了。vector扩容机制可以看前面的讲解  
![在这里插入图片描述](https://img-blog.csdnimg.cn/8f5fdd11e0e848d2aa9b77fc71591032.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### 一个deque自己会占用40字节大小；在G2.9中，deque可以自行指定缓冲区大小

这些大小的空间有：  
1.start，finish两个迭代器。分别指向第一缓冲区的第一个元素和最后缓冲区的最后一个元素（的下一位置）  
2.一个指向map的指针  
![在这里插入图片描述](https://img-blog.csdnimg.cn/99313f15fa774963ba46aa49d9619070.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.deque的insert操作解析

##### 判断顺序：

1.先是判断是否是头插或者尾插。是的话直接头尾插入元素即可。  
2.如果不是头插或者尾插，那么计算这个节点到头结点和尾节点之间的距离。假如说离头部节点近，那么就让从头部节点到插入位置之间的节点全部向前挪动，然后插入节点；反之亦然。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/d6a80e5578494739b0fe3c902729d3ab.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)  
![在这里插入图片描述](https://img-blog.csdnimg.cn/6b352b4553064212809e793e32fbbbfa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.deque模拟连续空间的手法

##### deque的长度需要考虑buffer的大小以及vector中buffer的个数

![在这里插入图片描述](https://img-blog.csdnimg.cn/ccefa143365b49bba898406e1e58bd4a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### deque的前进++和后退–操作需要额外判断是否超过当前buffer设定大小。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d758a7f050174cdaba9429d3c3ed6851.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### 如果有一次前进多个的情况，那么相较于上一种情况会更加复杂，需要考虑缓冲区之间的切换

![在这里插入图片描述](https://img-blog.csdnimg.cn/f9f2cea1eef34a2a8098458829b6d21e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 4.G4.9的deque

##### 在G4.9中，deque的大小没有变化。使用者无法再指定buffer size的大小。同时也变得更加复杂

![在这里插入图片描述](https://img-blog.csdnimg.cn/82ff41a95acd4ac093d7e781f5fcd026.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.queue和stack

#### 1.queue和stack的实现

对于queue和stack，它的功能只是deque的子集。实际上在STL中，queue和stack往往不被归类为容器，而被归类为**容器适配器**  
queue和stack中含有一个deque，然后调用已经完成的deque来完成我们需要的操作。

#### 2.queue和stack的异同

##### 相同：

1.不允许遍历，不提供迭代器  
2.可以使用deque或者list作为底层结构。不过一般使用deque，因为deque更加快  
3.不可以使用set或者map作为底层结构。

##### 不同：

queue不可以使用vector作为底层结构，而stack可以。

## 9.深度探索红黑树（RB-Tree），以及以它为基础的set，multiset，map，multimap

### 1.红黑树（RB-Tree）

这里不涉及红黑树的具体实现细节。红黑树的实现看此[浅析红黑树（RBTree）原理及实现](https://blog.csdn.net/tanrui519521/article/details/80980135)  
红黑树保持了BST的性质，但是同时保证最长路径不超过最短路径的二倍，因而近似平衡，避免树左右失衡导致插入和查找效率降低。  
因为红黑树是一颗BST，这样迭代器不应该修改那些已经排序插入的节点值。但是由于在C++中红黑树是作为set和map的底层，而map支持修改value，所以在C++中，红黑树**没有阻止我们去修改节点值**。  
红黑树对外界提供了两种插入形式，insert\_unique()和insert\_equal()，前者代表key在这颗红黑树中是唯一的，否则插入失败；而后者不是。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/1d3b9860a7ef4e3dbb32a07812490050.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

一个红黑树的例子如下：  
![在这里插入图片描述](https://img-blog.csdnimg.cn/6daedbf6980c45b3a9bc22b5dae503e2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.set 和 multiset

set和multiset其实也只是调用红黑树的部分函数，某种意义上它也只是一种适配器

#### 1.注意点：

1.set的key == value  
2.虽然set/multiset底层的红黑树支持修改节点值，但是set/multiset的迭代器并**不支持修改节点值**  
3.set和multiset的插入函数不相同  
4.虽然C++有全局泛化的::find()函数，但是它的效率远远不如set中定义的set::find()，我们**应当尽量优先使用容器中定义的函数**

![在这里插入图片描述](https://img-blog.csdnimg.cn/74da3d4102324394b43f7c7220da66d4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.set和multiset的实现

![在这里插入图片描述](https://img-blog.csdnimg.cn/f3bbcae26043426b972f08d70f1ba463.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.map 和 multimap

#### 1.注意点：

1.map的key != value  
2.map/multimap的迭代器禁止修改key，但是允许修改value  
3.map和multimap的插入函数不相同

#### 2.map和multimap的实现

![在这里插入图片描述](https://img-blog.csdnimg.cn/9ed963de5c6345c39d8f3cd10d1a98d4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.map和multimap独有的\[\]运算符设计

我们通过\[\]访问map/multimap，如果这个key不存在与map/multimap中，那么他会自动在map/multimap中创建并添加一个这个key对应的value的默认值，然后将其返回。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/e77f5d44ee8142bbb2fd19fb41e18042.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 11.深度探索哈希表（hashtable），以及以它为基础的unordered\_set，unordered\_multiset，unordered\_map，unordered\_multimap

### 1.哈希表（hashtable）

#### 1.基础概念

哈希表是为了实现高效的**存储**以及高效**查找**而实现的。具体操作就是将我们需要存放的数据进行哈希运算之后得到哈希值，然后将哈希值取模，插入哈希表中对应的篮子（basket）中去。  
哈希表的长度是一个**质数**；

##### Separate Chaining：当出现哈希碰撞时，将相同哈希值的节点组成一个链表挂在这个值对应的哈希值的后面。

##### Rehashing：当哈希表中的总元素数量 >= 哈希表长度时，将哈希表的长度扩展到它两倍原本大小的最近的质数（不是vector的两倍扩容，而是寻找离它两倍大小值最近的一个质数，作为新的大小），然后将元素重新插入。

![在这里插入图片描述](https://img-blog.csdnimg.cn/6a34863da1e545f3a4f7cb8a75ced8fa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.容器分析

哈希表需要以下6个模板参数：

|名称|作用|
|--|--|
|Value|和红黑树中一样，是键值的合成包|
|Key|键的类型|
|HashFcn|用来计算传入key的哈希值，得到hashcode，从而在哈希表中找到插入位置|
|ExtractKey|由于哈希表中存放元素也是key和value包，ExtractKey就是为了拿出这个包中的key值|
|EqualKey|告诉哈希表key“相等”的定义|
|Alloc|分配器|

![在这里插入图片描述](https://img-blog.csdnimg.cn/8df630e2ceda48cfbe89bfb6302a9451.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.哈希函数（HashFcn）

C++中为我们封装好了一些已有的哈希函数。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/8e1f40bb63a2434284d73a07d162114e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.无序容器(unordered容器)

和上面的有序容器相比，最大的区别就是底层实现变了。一个是红黑树，一个是哈希表。

### 3.（补充）红黑树实现和哈希表实现的这四个容器的区别

1.map始终保证遍历的时候是按key的大小顺序的，这是一个主要的功能上的差异。（有序无序）  
2.时间复杂度上，红黑树的插入删除查找性能都是O(logN)而哈希表的插入删除查找性能理论上都是O(1)，他是相对于稳定的，最差情况下都是高效的。哈希表的插入删除操作的理论上时间复杂度是常数时间的，这有个前提就是哈希表不发生数据碰撞。在发生碰撞的最坏的情况下，哈希表的插入和删除时间复杂度最坏能达到O(n)。注释：最坏情况就是所有的哈希值全部都在同一个链表上  
3.map可以做范围查找，而unordered\_map不可以。  
4.unordered\_map内存占用比map高。  
5\. 扩容导致迭代器失效。 map的iterator除非指向元素被删除，否则永远不会失效。unordered\_map的iterator在对unordered\_map修改时有时会失效。因为在操作 unordered\_map 容器过程（尤其是向容器中添加新键值对）中，一旦当前容器的负载因子超过最大负载因子（默认值为 1.0），该容器就会适当增加桶的数量（通常是翻一倍），并自动执行 rehash() 成员方法，重新调整各个键值对的存储位置（此过程又称“重哈希”），此过程很可能导致之前创建的迭代器失效。  
[出处1](https://zhuanlan.zhihu.com/p/358346216)  
[出处2](http://c.biancheng.net/view/7236.html)

## 12.算法

算法实际上看不到容器，它通过迭代器来进行运算。算法通过询问迭代器（之前有说迭代器需要提供的五个相关类型）来完成自己的工作。  
算法在语言层面是是一个函数模板，具体是一个[仿函数](https://so.csdn.net/so/search?q=%E4%BB%BF%E5%87%BD%E6%95%B0&spm=1001.2101.3001.7020)。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/7bef4cfd1ede4a959d8b41eb0d9aea0e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 1.迭代器

#### 1.各种容器的iterator\_category

##### 一共有五种iterator\_category，它们的关系如图所示

\[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-ftQ7gVZs-1643913242997)(en-resource://database/1125:1)\]

|iterator\_category|简述|容器|
|--|--|--|
|forward\_iterator\_tag|仅单向前进|forward\_list，unordered\_set，unordered\_map，unordered\_multiset，unordered\_multimap|
|bidirectional\_iterator\_tag|双向，允许前进和后退|list，set，map，multiset，multimap|
|random\_access\_iterator\_tag|允许访问随机下标|array，vector，deque|

另外有两种比较特殊，他们各自仅包含了一种迭代器

|iterator\_category|包含的迭代器|
|--|--|
|input\_iterator\_tag|istream\_iterator|
|output\_iterator\_tag|ostream\_iterator|

##### 通过typeid()可以获取iterator\_category

可以注意到这些打印出来的iterator\_category名称前后有一些无规律字符和数字，这些是编译器中的库实现方法决定的，编译器不同，这些数据也不同。但是实际上为了符合C++标准，它的实际类型都是一样的。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/6c1f013ded3c4f3ab24b8f8ad456e9a0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.迭代器对算法的影响的四个实例

##### distance

![在这里插入图片描述](https://img-blog.csdnimg.cn/7691c976df184dd6a22640b262f0357b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### advance(和distance近似，略)

和distance的做法基本相同。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/81e5504145334cc9802db3579186443c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### copy

copy用了很多次泛化和特化，除了iterator traits以外还用了type traits。  
copy对其template参数所要求的条件非常宽松。其输入区间只需由inputIterators构成即可，输出区间只需要由OutputIterator构成即可。这意味着可以使用copy算法，将任何容器的任何一段区间的内容，复制到任何容器的任何一段区间上  
![在这里插入图片描述](https://img-blog.csdnimg.cn/ca76b8fc172a4e438620f23ba736b6b7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### destory(和copy近似，略)

![在这里插入图片描述](https://img-blog.csdnimg.cn/d41ee1f89d6244a4bcbad938c4585a4f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.迭代器的特殊情况

![在这里插入图片描述](https://img-blog.csdnimg.cn/c62fbeeb169641b796dd0b7b70ab2353.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 4.算法对迭代器中iterator\_category的暗示

由于算法必须接受所有的迭代器，但是算法本身可以选择不对其进行处理。对于这些算法不想处理的迭代器，算法会在源代码中进行一些暗示。  
比如说这里就是特意修改了模板参数名，来暗示使用者这个算法的适用范围  
![在这里插入图片描述](https://img-blog.csdnimg.cn/818107a8b7f74042b9f9830f03682031.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.算法实例

![在这里插入图片描述](https://img-blog.csdnimg.cn/f44fd66a97bd46d380af6c96d10c5195.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### Q：如何判断是C的算法还是C++中的算法？

##### A：首先C++的算法应该在标准库std中，其次查看它的形参，C++需要满足接受接受至少两个参数来表示迭代器。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2fc87ff52a46435597ba27eeb2711481.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 十一个算法

![在这里插入图片描述](https://img-blog.csdnimg.cn/23c898068bdc4f76b1c57d5a34e8be04.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 13.仿函数functor

#### 1.functor简介

functor为算法服务，当算法需要一些额外准则时，我们使用仿函数来辅助实现算法。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/f7c4906234804ba8b9e032d9865a5924.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.让functor融入STL，允许被adapter改造

我们可以自行编写我们需要的functor，但是如果我们希望将它纳入STL，允许被adapter改造  
那就就必须遵循STL的规范，让它继承一些东西。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/9cfb0c66bdff476aa87f04a860cbadb2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 14.适配器adapter

### 1.适配器简介

适配器在STL组件的灵活组合运用功能上，扮演着轴承、转换器的角色  
STL所提供的各种适配器中：  
1）改变仿函数接口者，称为函数适配器；  
2）改变容器接口者，称为容器适配器；  
3）改变迭代器接口者，称为迭代器适配器  
对于函数适配器，适配器他也需要获得对应的仿函数一些信息。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/7705fff585814055916ac664e7bf294a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.容器适配器

STL提供两个容器适配器：queue和stack，它们修饰deque的接口而生成新的容器风貌stack的底层由deque构成。  
stack封锁住了所有的deque对外接口，只开放符合stack原则的几个函数  
queue的底层也由deque构成。queue封锁住了所有的deque对外接口，只开放符合queue原则的几个函数  
![在这里插入图片描述](https://img-blog.csdnimg.cn/01b06eaeae0147aa8fc73284f598feea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.函数适配器

![在这里插入图片描述](https://img-blog.csdnimg.cn/88cdbc4e6ee240cd9f110be318fb96c0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 1.bind2nd

##### 从bind2nd这个函数，我们可以看到函数适配器的一些巧妙之处

这里先复习一些前置知识  
对于模板，我们知道：  
1.对于类模板，它必须指明类中元素的类型，而不能由类自己推导  
2.对于函数模板，它有能力自己推导传入的参数类型。

```c_cpp
vector<int> vec;    //这个int表明我们必须声明类中元素类型
max(1,2);           //即使我们不声明参数1和2的类型，函数max也可以为我们自动推导出他们的类型。
12
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/e882f60adce74a12ba707cac1525532a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.not1

![在这里插入图片描述](https://img-blog.csdnimg.cn/ba5811f6d3cf4b0cbf352f4b28a32848.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 3.bind和占位符（C++ 11）

在C++11中，之前的适配器有一些被重新取代了，如图  
![在这里插入图片描述](https://img-blog.csdnimg.cn/a57bd4a460d54a53b3c940ae3e0a2e4b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 4.迭代器适配器

#### 1.reverse\_iterator

可以通过一个双向顺序容器调用rbegin()，和rend()来获取相应的逆向迭代器。只要双向顺序容器提供了begin(),end()，它的rbegin()和rend()就如同下面的形式。  
单向顺序容器slist不可使用reserve iterators。有些容器如stack、queue、priority\_queue并不提供begin()，end()，当然也就没有rbegin()和rend()  
![在这里插入图片描述](https://img-blog.csdnimg.cn/9c2c17947d6944ef9199b37bfaac69ad.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.insert\_iterator

insert iterators：可以将一般迭代的赋值操作转变为插入操作，可以分为下面几个  
insert iterators实现的主要观念是：每一个insert iterators内部都维护有一个容器（必须由用户指定）；容器当然有自己的迭代器，于是，当客户端对insert iterators做赋值操作时，就在insert iterators中被转为对该容器的迭代器做插入操作（也就是说，调用底层容器的push\_front()或push\_back()或insert()）

|insert iterator|作用|
|--|--|
|back\_insert\_iterator|专门负责尾端的插入操作|
|front\_insert\_iterator|专门负责首部的插入操作|
|insert\_iterator|可以从任意位置执行插入操作|

![在这里插入图片描述](https://img-blog.csdnimg.cn/b10d0dc4fd7349f89c6bf5dda5d0be63.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 5.未知适配器

iostream\_iterator严格来说不属于上面任何一种适配器，所以我们这里称之为“未知适配器”

#### 1.ostream\_iterator

![在这里插入图片描述](https://img-blog.csdnimg.cn/f0c7ca62339447c58bf51a4d8eb5d717.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.istream\_iterator

![在这里插入图片描述](https://img-blog.csdnimg.cn/4c52e15731224c199da11a7f498a14bf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 15.STL的周边技术与应用

### 1.一个万用的哈希运算

在哈希表一章中，我们有提到我们需要提供一个哈希运算方法来计算hashcode，并且这个哈希运算得到的结果应当尽可能无序。  
那么C++有没有自带这样一种函数，可以方便的为我们计算哈希值呢？  
我们可以粗略的这样进行一下思考：不管我们自己定义了什么类，这些类中的基本数据类型都是常见的，比如说int，string等等，如果可以挨个将其进行哈希运算，这个哈希运算式不就是一个万用的哈希运算了吗？下面就将介绍这种函数。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/f5e2ec3633524432b83084173ec6decc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 2.tuple

#### 1.tuple简介

tuple是C++11新标准里的类型。它是一个类似pair类型的模板。pair类型是每个成员变量各自可以是任意类型，但是只能有俩个成员，而tuple与pair不同的是它可以有任意数量的成员。但是每个确定的tuple类型的成员数目是固定的。

|操作|说明|
|--|--|
|make\_tuple(v1,v2,v3,v4…vn)|返回一个给定初始值初始化的tuple,类型从初始值推断|
|t1 == t2|2个tuple具有相同数量的成员且成员对应相等时返回true|
|get(t)|返回t的第i个数据成员|
|tuple\_size::value|tuple中成员的数量|
|tuple\_element::type|返回tuple中第i个元素的类型|

#### 2.tuple实现

tuple的关键就是利用C++的可变模板参数，来实现的的这个层层继承的关系。  
![在这里插入图片描述](https://img-blog.csdnimg.cn/380fab184cc84ead8fc86d300f496689.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.type traits

#### 1.type traits简介

在G2.9中，我们如果想要使用type traits，那么我们需要通过模板偏特化，然后typedef一大堆属性，用来保证以后算法来问的时候可以回答。  
问题可想而知，就是这些属性过多，写起来比较冗长  
![在这里插入图片描述](https://img-blog.csdnimg.cn/bb4ba44c11264594b912a3c24becfd70.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

注：POD类型是C++中常见的概念，用来说明类/结构体的属性，具体来说它是指没有使用面相对象的思想来设计的类/结构体。POD的全称是Plain Old Data，Plain表明它是一个普通的类型，没有虚函数虚继承等特性；Old表明它与C兼容。  
详细看此处[C++中的POD类型](https://zhuanlan.zhihu.com/p/56161728)

##### 在C++11中，这些type traits变得更加多，多达几十个，这样对于我们自己编写的类，要想使用type traits就变得更加冗长；但是在C++11中，不仅仅C++自带类可以自动提供自带的type traits，连我们自己编写的类都可以自动提供正确的type traits结果，不再需要我们自己编写，这是怎么实现的呢？

![在这里插入图片描述](https://img-blog.csdnimg.cn/e4fb1f8150d14af3b23cf89e5b70e823.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

#### 2.type traits实现

##### 对于一些简单的traits，可以找到源代码，是通过模板偏特化来实现的

![在这里插入图片描述](https://img-blog.csdnimg.cn/b0f89517d999434eb09b1b881f3c9d4d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### 然后对于一些复杂的type traits，无法在C++标准库中找到，猜测是编译器在运行期间推导出来的

![在这里插入图片描述](https://img-blog.csdnimg.cn/dc8b79e084af46319b4568436361affb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 4.cout

cout之所以可以接受那么多类对象，是因为标准对操作符<<做出了非常多的重载  
![在这里插入图片描述](https://img-blog.csdnimg.cn/0988d7050ee145a5a6e5fa883fa24da6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

如果我们自己编写的类的对象想要进行打印，就需要自己对<<进行重载  
![在这里插入图片描述](https://img-blog.csdnimg.cn/425dee30b38441c7ab3bc6a26ba97bb7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBASU5saW5LQw==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 5.std::move

[c++ 之 std::move 原理实现与用法总结](https://blog.csdn.net/p942005405/article/details/84644069)  
[c++ 左值引用与右值引用](https://zhuanlan.zhihu.com/p/97128024)

## Reference

1.[arkingc/note](https://github.com/arkingc/note/blob/master/C++/STL%E6%BA%90%E7%A0%81%E5%89%96%E6%9E%90.md)  
2.[C语言中文网](http://c.biancheng.net/)  
3.[超多电子书与视频资料分享](https://github.com/tangtangcoding/C-C-)  
4.[C++内存分配详解四：std::alloc行为剖析](https://blog.csdn.net/qq_34269632/article/details/115636008)  
5.[C++11新特性占位符-std::placehoders](https://blog.csdn.net/u014303647/article/details/88362337)  
6.[面试题：C++vector的动态扩容，为何是1.5倍或者是2倍](https://blog.csdn.net/qq_44918090/article/details/120583540)  
7.[C++虚函数表，虚表指针，内存分布](https://blog.csdn.net/li1914309758/article/details/79916414)  
8.[C++中tuple类型](https://www.cnblogs.com/huangfuyuan/p/9238598.html)  
9.[C++中的POD类型](https://zhuanlan.zhihu.com/p/56161728)  
10.[c++ 之 std::move 原理实现与用法总结](https://blog.csdn.net/p942005405/article/details/84644069)  
11.[c++ 左值引用与右值引用](https://zhuanlan.zhihu.com/p/97128024)  
12.[面试题：C++vector的动态扩容，为何是1.5倍或者是2倍？](https://blog.csdn.net/qq_44918090/article/details/120583540)
