# Cmake

## 一、Cmake安装

### Linux环境安装

1. 进入 [Cmake官网下载地址](https://cmake.org/download/  " 下载" ) 选择对应版本复制下载链接
2. 使用命令`wget https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz` 下载Cmake
3. 使用命令解压 `tar zxvf  cmake-3.21.0-linux-x86_64.tar.gz`
4. 在 `/opt` 中创建**cmake**目录，使用 `sudo mv cmake-3.21.0-linux-x86_64 /opt/cmake` 将解压的文件移动到**cmake**目录中
5. 使用命令 `ln -sf /opt/cmake/cmake-3.21.0-linux-x86_64/bin/* /usr/bin`,将 **Cmake的bin**链接到 **/usr/bin**
6. 使用命令 `cmake -version`测试

## 二、Cmake使用

### 单文件示例

1. 输出hello world的 main.cpp
   ```cpp
   // main.c
   #include <stdio.h>
   int main()
   {
   printf("hello world");
   return 0;
   }
   ```
2. 编写CMakeList.txt
   ```cmake
   cmake_minimum_required(VERSION 3.4.1)#最低版本要求
   project(demo CXX)#项目名字 以及使用的变成语言
   add_executable(demo main.cpp) # 生成名为demo的可执行文件。
   ```
3. 在CMakeList.txt所在目录使用 `cmake ./` 生成Makefile文件
4. 使用 `make`命令编译链接

### 多文件示例

1. 一个项目输出hello world，分别有hello.h、hello.cpp和main.cpp
   ```cpp
   hello.h 
   #ifndef TEST_HELLO_
   #define TEST_HELLO_
   void hello(const char* name);
   #endif //TEST_HELLO_
   
   hello.cpp
   #include <stdio.h>
   #include "hello.h"
   void hello(const char * name)
   {
   printf ("Hello %s!/n", name);
   }
   
   main.cpp
   #include "hello.h"
   int main()
   {
   hello("World");
   return 0;
   }
   ```
2. 编写CMakeList.txt
   ```cmake
   cmake_minimum_required(VERSION 3.4.1)#最低版本要求
   project(demo CXX)#项目名字 以及使用的变成语言
   set(SRC_LIST main.cpp hello.cpp)#设置需要编译的文件
   add_executable(demo ${SRC_LIST}) # 生成名为demo的可执行文件
   
   add_executable(demo main.cpp hello.cpp) # 生成名为demo的可执行文件，等同于上两个命令。
   
   ```
3. 在CMakeList.txt所在目录使用 `cmake ./` 生成Makefile文件
4. 使用 `make`命令编译链接

### 生成库使用

1. 一个项目输出hello world，分别有hello.h、hello.cpp和main.cpp
   ```cpp
   hello.h 
   #ifndef TEST_HELLO_
   #define TEST_HELLO_
   void hello(const char* name);
   #endif //TEST_HELLO_
   
   hello.cpp
   #include <stdio.h>
   #include "hello.h"
   void hello(const char * name)
   {
   printf ("Hello %s!/n", name);
   }
   
   main.cpp
   #include "hello.h"
   int main()
   {
   hello("World");
   return 0;
   }
   ```
2. 编写CMakeList.txt
   ```cmake
   cmake_minimum_required(VERSION 3.4.1)#最低版本要求
   project(demo CXX)#项目名字 以及使用的变成语言
   set(LIB_SRC hello.cpp)#设置需要生成库的文件
   set(SRC_LIST main.cpp hello.c)#设置需要编译的文件
   add_library(libhello ${LIB_SRC})#生成名为libhello的库文件
   add_executable(demo ${SRC_LIST}) # 生成名为demo的可执行文件
   ```
3. 在CMakeList.txt所在目录使用 `cmake ./` 生成Makefile文件
4. 使用 `make`命令编译链接

### 代码分开放设置CMakeList

1. 顶层的CMakeList.txt 文件 
   ```cmake
   project(HELLO)
   add_subdirectory(src)
   add_subdirectory(libhello)
   ```
2. src 中的 CMakeList.txt 文件
   ```
   include_directories(${PROJECT_SOURCE_DIR}/libhello)
   set(APP_SRC main.cpp)
   add_executable(hello ${APP_SRC})
   target_link_libraries(hello libhello)
   ```
3. libhello 中的 CMakeList.txt 文件
   ```
   set(LIB_SRC hello.cpp)
   add_library(libhello ${LIB_SRC})
   set_target_properties(libhello PROPERTIES OUTPUT_NAME "hello")
   ```
4. 建立一个build目录，在其内运行cmake，然后可以得到
build/src/hello.exe
build/libhello/hello.lib

## 常用命令

```cmake
本CMakeLists.txt所在的文件夹路径
${PROJECT_SOURCE_DIR}
#本CMakeLists.txt的project名称
${PROJECT_NAME} 

# 获取路径下所有的.cpp/.c/.cc文件，并赋值给变量中
aux_source_directory(路径 变量)
# 给文件名/路径名或其他字符串起别名，用${变量}获取变量内容
set(变量 文件名/路径/...)

# 添加编译选项
add_definitions(编译选项)

# 编译子文件夹的CMakeLists.txt
add_subdirectory(子文件夹名称)

# 设置可执行文件和库文件输出的目录
set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/lib)

# 将.cpp/.c/.cc文件生成.a静态库或动态库
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
add_library(库文件名称 STATIC/SHARED  文件)

# 将.cpp/.c/.cc文件生成可执行文件
add_executable(可执行文件名称 文件)

# 规定.h头文件路径
include_directories(路径)

# 规定.so/.a库文件路径
link_directories(路径)

# 对add_library或add_executable生成的文件进行链接操作
# 注意，库文件名称通常为libxxx.so，在这里只要写xxx即可
target_link_libraries(库文件名称/可执行文件名称 链接的库文件名称)
```

<br/>

切换cmake编译器

```
sudo update-alternatives --config cc
sudo update-alternatives --config c++
```