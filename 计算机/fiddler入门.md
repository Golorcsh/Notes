## 一、Fiddler简介

Fiddler是最强大最好用的Web调试工具之一， 它能记录所有客户端和服务器的http和https请求。允许你监视、设置断点、甚至修改输入输出数据。Fiddler包含了一个强大的基于事件脚本的子系统，并且能使用.net语言进行扩展。换言之，你对HTTP 协议越了解，你就能越掌握Fiddler的使用方法。你越使用Fiddler，就越能帮助你了解HTTP协议。Fiddler无论对开发人员或者测试人员来说，都是非常有用的工具。

## 二、Fiddler的工作原理

Fiddler 是以代理web服务器的形式工作的，它使用代理地址:127.0.0.1，端口:8888。 当Fiddler退出的时候它会自动注销， 这样就不会影响别的程序。不过如果Fiddler非正常退出，这时候因为Fiddler没有自动注销，会造成网页无法访问。 解决的办法是重新启动下Fiddler。

个人理解：fiddler是一个抓包工具，当浏览器访问服务器会形成一个请求，此时，fiddler就处于请求之间，当浏览器发送请求，会先经过fiddler，然后在到服务器；当服务器有返回数据给浏览器显示时，也会先经过fiddler，然后数据才到浏览器中显示，这样一个过程，fiddler就抓取到了请求和响应的整个过程。

正常退出方式：

[![](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923210954759-1901618191.png)](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923210954759-1901618191.png "")

### Fiddler界面

[![](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923211930353-1584573947.png)](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923211930353-1584573947.png "")

## 三、http协议介绍

协议是指计算机通信网络中两台计算机之间进行通信所必须共同遵守的规定或规则，超文本传输协议(HTTP)是一种通信协议，它允许将超文本标记语言(HTML)文档从Web服务器传送到客户端的浏览器。

### HTTP协议的主要特点

**1.支持客户/服务器模式** 
 **2.简单快速** ：客户向服务器请求服务时，只需传送请求方法和路径。 请求方法常用的有GET、HEAD、POST。 每种方法规定了客户与服务器联系的类型不同。由于HTTP协议简单， 使得HTTP服务器的程序规模小，因而通信速度很快。
 **3.灵活** ：HTTP允许传输任意类型的数据对象。正在传输的类型由Content-Type加以标记。
 **4.无连接** ： 无连接的含义是限制每次连接只处理一个请求。
服务器处理完客户的请求， 并收到客户的应答后， 即断开连接。 采用这种方式可以节省传输时间。
 **5.无状态** ：HTTP协议是无状态协议。无状态是指协议对于事务处理没有记忆能力。缺少状态意味着如果后续处理需要前面的信息，则它必须重传，这样可能导致每次连接传送的数据量增大。另一方面，在服务器不需要先前信息时它的应答就较快。

### HTTP协议之请求

http请求由三部分组成，分别是：**请求行、消息报头、请求正文**

请求方法有多种， 各个方法的解释如下：

**GET** 请求获取Request-URI所标识的资源

**POST** 在Request-URI所标识的资源后附加新的数据

**HEAD** 请求获取由Request-URI所标识的资源的响应消息报头

**PUT** 请求服务器存储一个资源， 并用Request-URI作为其标识

**DELETE** 请求服务器删除Request-URI所标识的资源

**TRACE** 请求服务器回送收到的请求信息，主要用于测试或诊断

**CONNECT** 保留将来使用

**OPTIONS** 请求查询服务器的性能，或者查询与资源相关的选项和需求

应用举例：

**GET方法** 在浏览器的地址栏中输入网址的方式访问网页时， 浏览器采用GET方法向服务器获取资源，eg:GET /form.html HTTP/1.1 (CRLF)

**POST方法** 要求被请求服务器接受附在请求后面的数据， 常用于提交表单。

### HTTP协议之响应

在接收和解释请求消息后，服务器返回一个HTTP响应消息。

HTTP响应也是由三个部分组成，分别是：**状态行、消息报头、响应正文**

状态代码有三位数字组成，第一个数字定义了响应的类别，且有五种可能取值：

1xx：指示信息--表示请求已接收，继续处理

2xx：成功--表示请求已被成功接收、理解、接受

3xx：重定向--要完成请求必须进行更进一步的操作

4xx：客户端错误--请求有语法错误或请求无法实现

5xx：服务器端错误--服务器未能实现合法的请求

常见状态代码、状态描述、说明：

200 OK //客户端请求成功

400 Bad Request //客户端请求有语法错误， 不能被服务器所理解

401 Unauthorized //请求未经授权，这个状态代码必须和WWW-Authenticate报头域一起使用

403 Forbidden //服务器收到请求，但是拒绝提供服务

404 Not Found //请求资源不存在，eg： 输入了错误的URL

500 Internal Server Error //服务器发生不可预期的错误

503 Server Unavailable //服务器当前不能处理客户端的请求，一段时间后可能恢复正常

## 四、Fiddler抓包解析

### 1.左侧面板

[![](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923213118337-1719287350.png)](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923213118337-1719287350.png "")

**抓包工具面板功能**

**#**  :HTTP Request的顺序，从1开始，按照页面加载请求的顺序递增。

**Result** : HTTP响应的状态 Protocol：请求使用的协议（如HTTP/HTTPS）

**HOST**：请求地址的域名 URL：请求的服务器路径和文件名，也包含GET参数

**BODY** ：请求的大小，以byte为单位

**Content-Type** ：请求响应的类型

**Caching** ：请求的缓存过期时间或缓存控制header的值

**Process**：发出此请求的Windows进程及进程ID

**Comments** ：用户通过脚本或者菜单给此session增加的备注

**custom**：用户可以通过脚本设置的自定义值

### 2.右侧面板

[![](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923213552556-1540264809.png)](https://images2017.cnblogs.com/blog/1242227/201709/1242227-20170923213552556-1540264809.png "")

**Statistics统计页签**

通过该页签， 用户可以通过选择多个会话来得来这几个会话的总的信息统计，比如多个请求和传输的字节数。选择第一个请求和最后一个请求， 可获得整个页面加载所消耗的总体时间。从条形图表中还可以分别出哪些请求耗时最多， 从而对页面的访问进行访问速度优化

**inspectors检查页签**

它提供headers、textview、hexview,Raw等多种方式查看单条http请求的请求报文的信息，它分为上下两部分：上部分为HTTP Request（请求）展示，下部分为HTTPResponse（响应）展示

**AutoResponse自动响应页签**

Fiddler最实用的功能， 它可以抓取在线页面保存到本地进行调试， 大大减少了在线调试的困难， 可以让我们修改服务器端返回的数据， 例如让返回都是HTTP404或者读取本地文件作为返回内容。

可设置打开某网页显示自己想要的内容，比如抓取百度链接，点击add rule，设置如下所示：

[![](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001121439059-694664172.png)](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001121439059-694664172.png "")

[![](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122207106-1320400732.png)](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122207106-1320400732.png "")

到百度页面刷新即可显示该图片

**composer构建页签**

支持手动构建和发送HTTP， HTTPS和FTP请求， 我们还可以从web session列表中拖曳session， 把它放到composer选项卡中， 当我们点击Execute按钮， 把请求发送到服务器端。操作如下图所示：

[![](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122408372-1438722917.png)](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122408372-1438722917.png "")

这样设置发送的请求，就不是浏览器发出的了，而是fiddler发出的，查看inspectors里面的信息便可看出

**log日志页签** ： 打印日志

**Filters过滤页签**

过滤器可以对左侧的数据流列表进行过滤， 我们可以标记、 修改或隐藏某些特征的数据流。

[![](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122950403-1965507111.png)](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001122950403-1965507111.png "")

**Timeline时间轴页签**

时间轴，也称为Fiddler的瀑布图，展示网络请求时间的功能。 每个网络请求都会经历域名解析、建立连接、发送请求、接受数据等阶段。把多个请求以时间作为 X 轴， 用图表的形式展现出来， 就形成了瀑布图。 在左侧会话窗口点击一个或多个（同时按下 Ctrl 键），Timeline 便会显示指定内容从服务端传输到客户端的时间

## 五、Fiddler命令行工具

Fiddler的左下角有一个命令行工具叫做QuickExec,允许你直接输入命令。

常见得命令有：

**help** ： 打开官方的使用页面介绍， 所有的命令都会列出来

**cls**  ： 清屏 (Ctrl+x 也可以清屏)

**select**  ： 选择会话的命令， 选择所有相应类型select image、select css、select html

**?sometext** ： 查找字符串并高亮显示查找到的会话列表的条目，？qq.com

**&gt;size** : 选择请求响应大小小于size字节的会话

**=status/=method/@host**:查找状态、方法、主机相对应的session会话，=504，=get，@www.qq.com

**quit**：退出fiddler

Bpafter，Bps, bpv, bpm, bpu这几个命令主要用于批量设置断点

Bpafter xxx: 中断 URL 包含指定字符的全部 session 响应

Bps xxx:中断 HTTP 响应状态为指定字符的全部 session 响应。

Bpv xxx:中断指定请求方式的全部 session 响应

Bpm xxx:中断指定请求方式的全部 session 响应，等同于bpv xxx

Bpu xxx:与bpafter类似。

## **六、Fiddler应用**

### 1.手机抓包

①启动Fiddler， 打开菜单栏中的 Tools &gt; Fiddler Options， 打开“FiddlerOptions” 对话框

②在“Fiddler Options”对话框切换到“Connections” 选项卡， 然后勾选“Allowromote computers to connect” 后面的复选框， 然后点击“OK” 按钮

③在本机命令行输入： ipconfig， 找到本机的ip地址。

④打开android设备的“设置” -&gt;“WLAN”，找到你要连接的网络，在上面长按，然后选择“修改网络”，弹出网络设置对话框，然后勾选“显示高级选项”（不同的手机，设置方法有所不同）

⑤在“代理” 后面的输入框选择“手动”，在“代理服务器主机名”后面的输入框输入电脑的ip地址，在“代理服务器端口”后面的输入框输入8888， 然后点击“保存” 按钮

⑥然后启动android设备中的浏览器，访问百度的首页，在fiddler中可以看到完成的请求和响应数据

[![](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001124822731-571080558.png)](https://images2017.cnblogs.com/blog/1242227/201710/1242227-20171001124822731-571080558.png "")

**备注** ： 如果是Android模拟器中ip要填写10.0.2.2，genymotion模拟器中ip要填写:10.0.3.2，手机实机中ip填电脑的ip，端口就是burp或者fiddler监听的端口 ，要处于同一网络下

### 2.过滤功能

①选择Filters页签，勾选use Filters勾选 Request Headers 中的 Hide if url contains 过滤项

②在里面输入：REGEX:(?insx)/[^\?/]*\.(css|ico|jpg|png|gif|bmp|wav)(\?.*)?$

REGEX: 表示启用正则表达式(?insx) 设置正则解释的规则，忽略大小写等。

此表达式表示过滤掉 url 中包括 css、ico、jpg 等后缀的请求

③勾选 Request Headers中的show only if URL contains，在里面输入

REGEX:(?insx).*\.?baidu.com/home.* 只显示： baidu.com/Home

Fiddler过滤指定域名

第二个选项是只监控以下网址，如只监控百度，在下面的输入框里填上www.baidu.com

“No Host Filter”不设置hosts过滤

“Hide The Following Hosts”隐藏过滤到的域名

“Show Only The Following Hosts”只显示过滤到的域名

“Flag The Following Hosts”标记过滤到的域名

**本文作者：**  [温一壶清酒](https://www.cnblogs.com/hong-fithing "") 

 **本文链接：**  <https://www.cnblogs.com/hong-fithing/p/7582947.html> 
