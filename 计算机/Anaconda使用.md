# Anaconda常用命令使用

## 查看已安装的软件包、虚拟环境、版本

查看已安装软件包`conda list`

查看虚拟环境`conda env list` 或`conda info --envs`

查看conda版本`conda --version`

查看python版本`python --version`

## 创建虚拟环境 name（指定python版本，如python=3.9）

`conda create -n name python=3.9`

## 激活虚拟环境

　`conda activate xxx`　　 # for windows

　`source conda activate xxx`　　# for Linux & Mac

## 关闭虚拟环境

`conda deactivate`　　　　　 # for windows

`source conda deactivate`　　# for Linux & Mac

## 删除虚拟环境

`conda remove --name py35 --all`

## 换源

[清华大学源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)

若出现**conda连接网络出现错误(CondaHTTPError: HTTP 000 CONNECTION FAILED for url）**使用修改下方源

```
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
```