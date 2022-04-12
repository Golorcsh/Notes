# Gtest使用

## 一、安装

### 工具简介

`googletest`是`Google`公司开发一款跨平台（`Linux`、`Windows`、`Mac`）测试工具。

### 依赖说明

- `Bazel`或者 `Cmake`：文章采用`cmake`构建，官方推荐`Bazel`。
- 支持`C++11`标准的编译器：`GNU`的`c++`。

### 安装

```powershell
# 下载googletest源码
$ git clone https://github.com/google/googletest.git
# 进入/创建编译目录
$ cd googletest
$ mkdir build
$ cd build
# 编译 - cmake
$ cmake ../
$ make
#默认安装
$ sudo make install
```

执行编译后会得到四个静态库文件：

- `libgtest.a`
- `libgtest_main.a`
- `libmock.a`
- `libmock_main.a`

必要的头文件：

- `googltest/include/gtest`
- `googlemock/include/gmock`

手动安装过程：

- 拷贝静态库到自定义文件夹
- 拷贝头文件到自定义文件夹

默认安装（`make install`）：建议使用

- 静态库文件会被拷贝到`/usr/local/lib64/`
- 头文件会被拷贝到`/usr/local/include/`

## 二、使用

### 方法一

在安装好后，导入头文件`#include <gtest/gtest.h>`

```c_cpp
/* file:test.cc */
#include <stdio.h>
#include <gtest/gtest.h>

int add(int a, int b) {
    return a+b;
}

TEST(MyTest, AddTest) {
    EXPECT_EQ(add(1, 2), 3);
}

int main(int argc, char *argv[]) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
```

### 方法二

在没有安装gtest的情况下，直接将gtest项目包括在自己的项目中，则需在cmakelists.txt中增加下方设置

```
#googletest 路径添加
set(googleTestDir googletest)
#Add the google test subdirectory
add_subdirectory(${googleTestDir})
#include googletest/include dir
include_directories(${googleTestDir}/googletest/include)
#include the googlemock/include dir
include_directories(${googleTestDir}/googlemock/include)

#创建可执行文件
add_executable(demo main.cpp)

#链接googletest模块
target_link_libraries(demo gtest gmock gtest_main)
```
