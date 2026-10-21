# 四模型错误案例 Error Analysis

## 范围与口径

本报告只分析四个模型人工筛选后的 error/错误图片，共 370 张：ChatGPT 64 张、Doubao 100 张、Gemini 108 张、Qwen 98 张。未读取正确图片、未重判 Correct/Error，也未重新计算准确率。本次仅补看 Doubao #169 马格纳并追加错误类型记录。

每张图片只统计一个 Primary Error Pattern；Secondary Error 仅用于解释。所有角色、Tier、原子特征与判错红线均来自当前 `200角色看图打分工作簿.xlsx` 的“看图清单”。`error_pattern_summary.csv` 中 `Tier=All` 为模型总体统计，Tier 1–4 行为该 Tier 的 Primary Pattern 分解；Percentage 的分母始终是该模型 error 文件夹总图片数。

> 注意：下文比例描述的是“已筛选错误样本内部的构成”，不是四模型在全部 200 张样本上的错误率，因此不能据此重算或替代准确率。

## 模型内总结

### ChatGPT

- 错误图片总数：64
- Top 3 Primary Patterns：P2 Signature Feature Missing — 30（46.88%）；P9 Character Genericization — 9（14.06%）；P4 Costume / Prop Error — 8（12.50%）
- 错误样本中数量最多的 Tier：Tier 2 — 30 张（46.88%）
- 错误高度集中于 P2：许多图能保留角色的大轮廓，却遗漏疤痕、眼罩、特殊瞳孔、眼镜等小而决定身份的原子特征。
- Tier 3 中 P9 与 P10 同时突出，说明长尾角色即使服装或配色接近，也容易被拉回标准动漫脸和通用气质。
- P4 主要表现为武器、制服徽章或特殊道具结构缺失；Tier 4 的 6 张错误中，P11/P12 合计 5 张，复杂条件容易被省略或互相覆盖。

### Doubao

- 错误图片总数：100
- Top 3 Primary Patterns：P2 Signature Feature Missing — 23（23.00%）；P4 Costume / Prop Error — 18（18.00%）；P6 Spatial / Left-Right Error — 11（11.00%）
- 错误样本中数量最多的 Tier：Tier 2 — 53 张（53.00%）
- P2 与 P4 是主体，但分布比 ChatGPT 更分散，说明它既会漏身份细节，也会把武器、眼镜、制服等实体生成成错误类别。
- P6 与 P7 合计 20 张（20.00%），常见机制是左右镜像、纹样串位、局部特征被字面化到错误对象。
- Tier 4 错误共 16 张，其中 P11 有 9 张；它通常能画出部分概念，但容易漏掉某项明确材质、动作、环境或道具条件。
- 结构错误数量不算最高，更多问题出在元素之间的绑定关系与条件覆盖完整度。
- 补录马格纳后，Tier 3 错误为 22 张，P10 总体增至 7 张（7.00%）；新增案例体现特殊发型几何结构与硬分色退化，不改变前三类排名。

### Gemini

- 错误图片总数：108
- Top 3 Primary Patterns：P2 Signature Feature Missing — 28（25.93%）；P4 Costume / Prop Error — 19（17.59%）；P6 Spatial / Left-Right Error — 13（12.04%）
- 错误样本中数量最多的 Tier：Tier 2 — 53 张（49.07%）
- P2、P4、P6 构成主要错误：身份细节遗漏、道具结构错误和非对称特征镜像反复出现。
- P8 有 7 张（6.48%），为四模型中占比最高，集中在多手、多导管、虫群、链饰等高密度元素的粘连或噪点化。
- Tier 3 有 34 张错误，P9/P10 合计 16 张，特殊画风、非标准脸型和长尾角色经常被美型化或普通化。
- Tier 4 以 P11 为主，往往保留视觉上最强的概念，却遗漏书本、武器、身份发型等弱条件。

### Qwen

- 错误图片总数：98
- Top 3 Primary Patterns：P2 Signature Feature Missing — 20（20.41%）；P4 Costume / Prop Error — 17（17.35%）；P6 Spatial / Left-Right Error — 14（14.29%）
- 错误样本中数量最多的 Tier：Tier 2 — 45 张（45.92%）
- P2、P4、P6 是前三类，但 P1 达 9 张（9.18%），为四模型最高，出现角色性别改变、人物被动物化或原创设定完全替代原角色等身份级失败。
- P6 与 P7 合计 21 张（21.43%），同样是四模型最高，左右标记、武器归属、纹样所在对象和动作起点尤其容易绑定错误。
- Tier 3 的 34 张错误中，P1/P9/P10 合计 19 张（55.88%），长尾身份与特殊视觉语言保持最弱。
- P12 占 4.08%，整体占比最高；复杂组合中较常出现字面化、概念机械拼接或强设定吞没角色。

## 四模型横向比较

| 比较问题 | 结论（按已筛选错误样本构成） | 证据 | 
|---|---|---|
| 最容易出现 Character Identity Error | Qwen | P1：9/98（9.18%），四模型最高。 |
| 最容易遗漏 Signature Feature | ChatGPT | P2：30/64（46.88%），集中在疤痕、眼部标记、眼镜与小型配饰。 |
| 最容易发生位置/左右/属性错配 | Qwen，Doubao 接近 | Qwen P6+P7：21/98（21.43%）；Doubao：20/100（20.00%）。 |
| 结构生成问题最多 | Gemini | P8：7/108（6.48%），主要是多肢体、导管、链饰和群体元素粘连。 |
| 最容易 Character Genericization | ChatGPT | P9：9/64（14.06%），占比最高。 |
| Tier 3 长尾角色保持最差 | Qwen | 在其 Tier 3 错误中，P1+P9+P10 为 19/34（55.88%）；此处衡量的是错误样本内身份/普通化/退化集中度。 |
| Tier 4 最容易 Prompt Condition Omission | Doubao | P11：9/100（9.00%）；在其 Tier 4 错误中为 9/16（56.25%）。 |
| Multi-concept Fusion Failure 最严重 | Qwen | P12：4/98（4.08%），四模型整体占比最高。 |

### 系统性 failure pattern

- ChatGPT 的错误最集中：先保住角色大类，再漏掉决定身份的小型原子特征，并在 Tier 3 把角色拉回标准动漫脸。 
- Doubao 更像“元素覆盖不全或关系绑定不稳”：能生成许多组件，但会漏条件、镜像左右或把纹样/属性放错对象。 
- Gemini 在高密度、多部件图像中更易出现结构粘连和噪点化，同时对特殊画风也有明显美型化倾向。 
- Qwen 的身份级漂移和关系错配最突出，且复杂概念更容易被字面化或互相吞没；其错误机制比单纯漏细节更深。 

因此，四模型确实形成不同且可重复的 failure patterns：ChatGPT 偏“关键细节遗漏/普通化”，Doubao 偏“条件遗漏/关系错配”，Gemini 偏“复杂结构粘连/特殊画风退化”，Qwen 偏“身份漂移/属性绑定与融合失败”。
