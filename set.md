# **关闭oneDNN提示**
```
    set TF_ENABLE_ONEDNN_OPTS=0
    $env:TF_ENABLE_ONEDNN_OPTS = "0"
```
# **训练前修改MoPKL网络维度**
MoPKL.py修改网络中维度 text_input_dim
- ITSDT: 130*300
- DAUB-R: 20*300
- IRDST-H: 20*300
        