# Git Script

### proxy

```
# 
git config --global http.proxy http://<proxy_ip>:<proxy_port>
git config --global https.proxy https://<proxy_ip>:<proxy_port>

# 
git config --global --unset http.proxy
git config --global --unset https.proxy


```

### submodules

```
# add submodule
git submodule add <repository_url> <path>

# update submodule
git submodule update --init --recursive

# remove submodule
git rm --cached <path>
rm -rf .git/modules/<path>
```


