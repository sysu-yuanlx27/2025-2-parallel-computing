# 并行程序设计与算法

中山大学[并行程序设计与算法课程](https://easyhpc.net/course/258)作业源码仓库．

## 环境配置

本小节适用于各 Arch 系发行版．笔者的开发环境为 Arch Linux WSL．

```sh
sudo pacman -S openmpi intel-oneapi-mkl
```

实验报告使用 LaTeX 编写．

```sh
sudo pacman -S texlive texlive-langchinese
```

# 一键打包

以实验一为例，在 Shell 中执行：

```sh
git archive -o 01.zip HEAD lab/01
```

便会将最新一次 Commit 后 `lab/01` 文件夹中不在 `.gitignore` 中的所有文件打包成一个 `01.zip` 文件．