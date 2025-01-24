# Conda Scripts

### proxy

```cmd
# 查看代理
conda config --show proxy_servers
## proxy_servers: {}
## 或者
## proxy_servers:
##  http: http://192.168.2.3:7890
##  https: http://192.168.2.3:7890

# 设置代理
conda config --set proxy_servers.https http://127.0.0.1:7890
conda config --set proxy_servers.http http://127.0.0.1:7890

# 取消代理
conda config --remove-key proxy_servers.http
conda config --remove-key proxy_servers.https
```

## env

```
# create an environment named "name" with python3.8
conda create -n name python=3.8
# remove 
conda remove -n name --all
```

