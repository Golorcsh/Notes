# 基于Python语言的Selenium工具使用

## 一、爬虫(Crawler）

### 爬虫简介

- 爬虫又称为网络爬虫（又称为网页蜘蛛，网络机器人，更经常的称为网页追逐者），是一种按照一定的规则，自动地抓取互联网信息的程序或者脚本。

### 使用场景

- 搜索引擎
- 大数据处理（数据获取）
- 网络监控
- 等等

### 案例

- 搜索引擎：如百度、google等搜索引擎，使用爬虫，爬取互联网数据，然后整理归档。再将数据提供给用户。

## selenium工具

### 介绍

- Selenium 是一个用于Web应用程序测试的工具。Selenium测试直接运行在浏览器中，**就像真正的用户在操作一样**。支持的浏览器包括IE（7, 8, 9, 10, 11），Mozilla Firefox，Safari，Google Chrome，Opera等。
![selenium](images/教育技术/selenium与驱动的关系.png)
  
  ### 安装
- selnium安装
  - 在已安装python的电脑中，使用命令：
  `pip install selenium`
- 浏览器驱动安装
  - 根据使用的浏览选择对应的驱动和版本
  - 例如选择chrome浏览器，先在浏览器地址栏中输入：chrome://version/，查看浏览器版本。然后到网址：[谷歌浏览器chrome驱动](https://sites.google.com/a/chromium.org/chromedriver/home) 
  - 下载对于版本的驱动。下载后将驱动放入浏览器安装位置然后将路径添加到系统环境变量中方便使用，或者不添加路径，在编写代码时手动添加驱动路径。

### 使用

- 代码编写模块中具体展开

## 网页分析

### 网页元素的属性

- 元素class name
- 元素id
- 元素text文本
- 元素link链接
- tag标签
- 。。。

### 如何选择合适的属性来筛选元素

- `fin_element(s)_by_id`
- `fin_element(s)_by_name`
- `fin_element(s)_by_class_name`
- `fin_element(s)_by_link_text`
- `fin_element(s)_by_css_selector`
- `fin_element(s)_by_xpath`
- `fin_element(s)_by_tag_name`

## 爬虫编写    

### 编写代码

```
 from selenium import webdriver //导入工具类
 driver = webdriver.Chrome(）//将浏览器初始化
 driver.get('url')//打开网页
 driver.find_element_by_id('id属性')//通过id筛选网页元素
```
