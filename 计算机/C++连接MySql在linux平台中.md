# C++连接MySql在Linux中

## 1. 安装所需要的开发包(ubuntu)

在linux中输入`sudo apt-get install libmysqld-dev`

## 2.常用函数

|**函数**|**说明**|
|--|--|
|`int mysql_library_init(int argc, char \**argv, char \**groups)`|在调用任何其他MySQL函数之前，调用此函数初始化MySQL客户端库。调用mysql_init()时会自动调用|
|`MYSQL *mysql_init(MYSQL *mysql)`|获取或初始化MYSQL结构|
|`MYSQL *mysql_real_connect(MYSQL *mysql, const char *host, const char *user, const char *passwd, const char *db, unsigned int port, const char *unix_socket, unsigned long client_flag)`|连接到MySQL服务器。|
|`int mysql_query(MYSQL *mysql, const char *stmt_str)`|执行指定为“以Null终结的字符串”的SQL查询。|
|`MYSQL_RES *mysql_store_result(MYSQL *mysql)`|将查询的整个结果读取到客户端，分配一个 MYSQL_RES结构，并将结果放入此结构中|
|`unsigned int mysql_field_count(MYSQL *mysql)`|返回上次执行语句的结果集的列数。|
|`unsigned int mysql_num_fields(MYSQL_RES *result)`|返回结果集中的列数。|
|`my_ulonglong mysql_num_rows(MYSQL_RES *result)`|返回结果集中的行数。|
|`MYSQL_ROW mysql_fetch_row(MYSQL_RES *result)`|从结果集中获取下一行数据|
|`void mysql_free_result(MYSQL_RES *result)`|释放结果集空间|
|`void mysql_library_end(void)`|完成使用库后调用它（例如，在断开与服务器的连接后）|

### 相关数据结构

- MYSQL结构体
此结构表示一个数据库连接的处理程序。它几乎用于所有MySQL功能。
- MYSQL_RES结构体
存储查询结果数据。
- MYSQL_ROW结构体
存储一行数据的结构。

## 3.基本使用流程

1. 使用`mysql_init()`初始化连接
2. 使用`mysql_real_connect()`建立一个到mysql数据库的连接
3. 使用`mysql_query()`执行查询语句
4. `result = mysql_store_result(mysql)`获取结果集
5. `mysql_num_fields(result)`获取查询的列数，`mysql_num_rows(result)`获取结果集的行数
6. 通过`mysql_fetch_row(result)`不断获取下一行，然后循环输出
7. 释放结果集所占内存`mysql_free_result(result)`
8. `mysql_close(conn)`关闭连接
9. `void mysql_library_end(void)`  完成使用库后调用它（例如，在断开与服务器的连接后）
