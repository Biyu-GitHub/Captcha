# 基于深度学习的验证码识别

## 环境要求

* python 3.6
* captcha 0.3

## 快速开始

### 生成数据集

在指定的path目录下生成10k张验证码托图片，并划分出20%用作验证集：

```bash
# from Captcha/
python prepare_dataset.py --path=./images
```

生成的数据集的目录结构：

```
|-- Captcha/
	|-- images/
		|-- train/
		|-- test/
```

### 训练

### 测试

