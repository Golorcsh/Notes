# 1.常用请求

1. `curl -XPOST url` 使用post请求
2. `curl -XGET url` 使用get请求

curl默认使用get请求

# 2.文件下载

1. `curl -O url` 下载文件
2. `curl -o name url`  指定下载文件名
3. `curl --limit-rate 100k url` 设置下载速度
4. `curl -C - url`  恢复下载

# 3.连接和测试

1. `curl -v -L url` 查看信息

# 4. 其他

```
curl -u username:passwd -O | -o ftp_url 从ftp服务器下载文件
curl -u username:passwd -T file ftp_url 上传文件到ftp服务器
```
