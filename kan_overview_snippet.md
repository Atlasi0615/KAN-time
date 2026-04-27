
## 可直接加入文档的图片引用

生成总览图后，把下面这几段插入到 Markdown 里即可：

```markdown
### 12.2 原始 KAN 总览图

![](outputs_final/kan/20260423_210939_random/kan_specific/original_plot_overview.png)

### 12.3 稀疏化 KAN 总览图

![](outputs_final/kan/20260423_210939_random/kan_specific/sparse_plot_overview.png)

### 12.4 剪枝后 KAN 总览图

![](outputs_final/kan/20260423_210939_random/kan_specific/pruned_plot_overview.png)
```

## 运行命令

```bash
python scripts/make_kan_overview.py --kan-specific-dir outputs_final/kan/20260423_210939_random/kan_specific
```
