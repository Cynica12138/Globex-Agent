# 18-3 Prompt 自进化与版本化 A/B 测试

## 本章课程目标

- 理解为什么 Prompt 需要自进化——工具集变了、新 bad case 类型出现了、模型升级了，旧 prompt 就“过期”了。
- 掌握 Prompt 版本管理的工程实现：Git-like 语义化版本号 + changelog + 回滚能力。
- 掌握 Prompt A/B 测试框架：按 user_id hash 分流 + Rubric 对比 + 自动放量/回滚标准。
- 理解 Auto-Prompt-Optimization 的三阶段进化：手动 → 半自动 → 全自动。
- 看清 Prompt 变更对 Cache Breakpoint 的影响和 graceful migration 策略。

## 学习建议

这是进化闭环里最“轻”最“快”的路径——不需要训模型，不需要改代码，只改 Prompt 就能修复 30% 的退化问题（见 18-1 §4 的 80/20 法则）。

代价极低、见效极快。如果你只有时间做一件自进化相关的事，做这一件的 ROI 最高。

---

## 1、为什么 Prompt 需要自进化

### 1.1 Prompt 会“过期”的三个场景

| 场景 | 具体表现 | 如果不改会怎样 |
|---|---|---|
| **工具集变了** | 新增了 WebSearch 工具但 prompt 里没提及 | Agent 永远不知道可以搜外部资料 |
| **新 bad case 类型出现** | 用户开始问“跨境直邮免税额度”但 prompt 没有相关规则 | Agent 回答不了这类问题 |
| **模型版本升级** | 新模型对旧 prompt 的措辞理解不同（如 temperature 含义变了） | 同样的 prompt 产出不同行为 |

### 1.2 手动改 Prompt 的问题

```text
发现 bad case → 运营说“加一条规则” → 直接改 prompts.yml → 部署
  → 问题 1：不知道这次改动影响了多少 query（可能修了 A 又坏了 B）
  → 问题 2：没法回滚（改了就改了，旧版本没了）
  → 问题 3：多人同时改 prompt 互相冲突
  → 问题 4：改了 prompt 缓存命中率骤降（前缀变了）
```

**Prompt 自进化 = 把“随手改”升级为“有版本、有测试、有回滚、能自动”。**

---

## 2、Prompt 版本管理

### 2.1 语义化版本号

```text
格式：v{major}.{minor}.{patch}

major：范式级别变更（如 AgentLoop 循环结构改了）→ 必须全量重测
minor：新增规则 / 工具描述（如加了 WebSearch 描述）→ A/B 测试后放量
patch：措辞微调 / 修错别字（如“旅行三件套”改成“旅行套装”）→ 直接上线
```

### 2.2 版本存储

```python
# app/evolution/prompt_versions.py
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import yaml


@dataclass
class PromptVersion:
    version: str                     # "v1.3.2"
    content: str                     # 完整 prompt 文本
    changelog: str                   # 这次改了什么
    author: str                      # 谁改的 / "auto" 表示自动生成
    created_at: datetime = field(default_factory=datetime.now)
    rubric_score: float | None = None  # A/B 测试后的分数
    status: str = "draft"            # draft / testing / active / retired


class PromptVersionStore:
    """Prompt 版本管理器。"""

    def __init__(self, store_path: Path = Path("data/prompt_versions")):
        self._path = store_path
        self._path.mkdir(parents=True, exist_ok=True)

    def save(self, version: PromptVersion) -> None:
        filepath = self._path / f"{version.version}.yml"
        filepath.write_text(
            yaml.dump(version.__dict__, allow_unicode=True)
        )

    def get_active(self) -> PromptVersion:
        """获取当前生效的版本。"""
        for f in sorted(self._path.glob("*.yml"), reverse=True):
            v = self._load(f)
            if v.status == "active":
                return v
        raise RuntimeError("No active prompt version found")

    def get_version(self, version: str) -> PromptVersion:
        filepath = self._path / f"{version}.yml"
        return self._load(filepath)

    def rollback(self, to_version: str) -> None:
        """回滚：把 to_version 设为 active，当前 active 设为 retired。"""
        current = self.get_active()
        current.status = "retired"
        self.save(current)

        target = self.get_version(to_version)
        target.status = "active"
        self.save(target)

    def _load(self, path: Path) -> PromptVersion:
        data = yaml.safe_load(path.read_text())
        return PromptVersion(**data)


prompt_store = PromptVersionStore()
```

### 2.3 版本 changelog 示例

```yaml
# data/prompt_versions/v1.3.0.yml
version: "v1.3.0"
content: |
  你是 Globex 购物 Agent...
  # 新增：WebSearch 工具描述
  - WebSearch: 检索博主推荐、价格趋势等外部资料
  ...
changelog: |
  新增 WebSearch 工具描述（新接入的工具）
  修改 fork 三件事判断的措辞（更明确“上下文隔离”的含义）
author: "haojun"
created_at: "2026-07-01T10:00:00"
rubric_score: 0.81
status: "active"
```

---

## 3、Prompt A/B 测试框架

### 3.1 为什么需要 A/B 测试

改 Prompt 最怕的是“修了 A 坏了 B”——加了一条规则让某类 query 好了，但另一类 query 因为 prompt 变长导致 context rot，反而变差。

**A/B 测试的价值：在全量上线前用 10% 流量验证“新 prompt 整体上没有比旧的差”。**

### 3.2 分流实现

```python
# app/evolution/ab_router.py
import hashlib
from app.evolution.prompt_versions import prompt_store, PromptVersion


# A/B 测试配置
AB_TEST_RATIO = 0.10  # 10% 流量走新版本
_testing_version: PromptVersion | None = None


def get_prompt_for_user(user_id: str) -> str:
    """根据 user_id 决定用哪个版本的 prompt。"""
    global _testing_version

    if _testing_version is None or _testing_version.status != "testing":
        # 没有 A/B 测试在跑，直接用 active 版本
        return prompt_store.get_active().content

    # 按 user_id hash 分流
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    ratio = (hash_val % 100) / 100.0

    if ratio < AB_TEST_RATIO:
        return _testing_version.content  # 新版本
    else:
        return prompt_store.get_active().content  # 旧版本


def start_ab_test(new_version: PromptVersion):
    """启动 A/B 测试。"""
    global _testing_version
    new_version.status = "testing"
    prompt_store.save(new_version)
    _testing_version = new_version


def conclude_ab_test(accept: bool):
    """结束 A/B 测试。"""
    global _testing_version
    if _testing_version is None:
        return

    if accept:
        # 新版本胜出 → 设为 active
        old = prompt_store.get_active()
        old.status = "retired"
        prompt_store.save(old)
        _testing_version.status = "active"
    else:
        # 新版本不行 → 退回 draft
        _testing_version.status = "retired"

    prompt_store.save(_testing_version)
    _testing_version = None
```

### 3.3 A/B 评判标准

| 维度 | 判断条件 | 动作 |
|---|---|---|
| Rubric 均分 | 新 >= 旧 - 0.02（连续 3 天） | 放量 |
| Rubric 均分 | 新 < 旧 - 0.05（任意 1 天） | 自动回滚 |
| 格式正确率 | 新 < 96% | 自动回滚 |
| Cache 命中率 | 新 < 旧 × 0.7（说明前缀变化太大） | 告警 + 人工确认 |

### 3.4 自动放量策略

```text
Day 1-3：10% 流量 → 收集 Rubric 分
Day 4：如果满足“新 >= 旧 - 0.02” → 放到 30%
Day 5-6：30% 流量观察
Day 7：如果仍然满足 → 放到 100%（新版本变为 active）
```

---

## 4、Auto-Prompt-Optimization：从手动到全自动

### 4.1 三阶段进化

| 阶段 | 做法 | 人工参与程度 | 生效速度 |
|---|---|---:|---|
| 手动 | 运营看 bad case → 人工写规则 → 直接改 prompt | 100% | 分钟级 |
| 半自动 | LLM 分析 bad case → 生成 prompt 修改建议 → 人工审核后 A/B | ~30% | 天级 |
| 全自动 | LLM 分析 → 自动生成 → 自动 A/B → 自动合入 | 0% | 天级 |

### 4.2 半自动阶段的实现

```python
# app/evolution/auto_prompt.py
from app.agent.llm import get_judge_llm
from app.evolution.prompt_versions import PromptVersion, prompt_store


ANALYZE_PROMPT = """你是 Globex 的 Prompt 优化器。

当前 system prompt：
{current_prompt}

最近 5 条 Rubric 分 < 0.65 的 bad case 摘要：
{bad_cases_summary}

请分析这些 bad case 的共性问题，并给出 system prompt 的修改建议。
输出格式：
1. 问题诊断：一句话说清楚根因
2. 修改建议：给出要在 prompt 里加/改/删的具体文本
3. 预期效果：改完后预计对哪类 query 有帮助
"""


async def suggest_prompt_improvement(bad_cases: list[dict]) -> dict:
    """让强模型分析 bad case 并生成 prompt 修改建议。"""
    current = prompt_store.get_active()
    summary = "\n".join(
        f"- query: {c['query'][:50]}... | 扣分原因: {c['rubric_comment']}"
        for c in bad_cases[:5]
    )

    llm = get_judge_llm()
    resp = await llm.ainvoke([
        (
            "user",
            ANALYZE_PROMPT.format(
                current_prompt=current.content[:2000],
                bad_cases_summary=summary,
            ),
        ),
    ])

    return {
        "suggestion": resp.content,
        "based_on_version": current.version,
        "bad_cases_count": len(bad_cases),
    }
```

### 4.3 全自动阶段的完整流程

```text
每日凌晨自动执行：
  Step 1：从 18-2 章的 P1 bad case 池中取最近 7 天的 case
  Step 2：调 suggest_prompt_improvement() 生成修改建议
  Step 3：自动应用修改 → 生成新 PromptVersion（版本号 patch+1）
  Step 4：自动启动 A/B 测试（10% 流量）
  Step 5：3 天后自动评判：
    → 满足标准 → 自动放量到 100%
    → 不满足 → 自动回滚 + 标记这次尝试失败
```

### 4.4 全自动的安全门禁

| 门禁 | 条件 | 不通过时怎么办 |
|---|---|---|
| 修改幅度限制 | 新 prompt 和旧 prompt 的 diff 不超过 200 字 | 拒绝自动合入，推给人工 |
| 工具描述不能删 | 自动优化不能删除任何工具的描述 | 拒绝 |
| Fork 规则不能改 | Fork 三件事判断的措辞不能被自动修改 | 拒绝 |
| 日均改动上限 | 每天最多自动生成 1 个新版本 | 排队到明天 |

**原则：全自动只能做“加规则 / 改措辞”，不能做“删规则 / 改架构”——后者必须人工。**

---

## 5、Prompt 变更对 Cache Breakpoint 的影响

### 5.1 为什么 Prompt 变更会打崩缓存

第 5 章讲过：Prompt Cache 基于前缀匹配——只有消息序列的前缀和上次一样，才能命中。

System Prompt 是消息序列的**第一条**——它变了 = 整个前缀变了 = 所有缓存全部失效。

```text
旧 prompt 的请求：cache hit（前缀匹配成功）
新 prompt 的请求：cache miss（前缀变了）

如果 A/B 测试期间 10% 流量走新 prompt：
  → 这 10% 的请求 cache 全部 miss
  → token 成本增加
  → 如果 A/B 期间成本飙升，可能被误判为“新 prompt 不好”
```

### 5.2 Graceful Migration 策略

```python
# app/evolution/cache_migration.py

def migrate_prompt_gracefully(old_version: str, new_version: str):
    """渐进式迁移 prompt，减少缓存失效冲击。"""

    # 策略 1：A/B 测试期间不统计 token 成本对比
    # （因为新 prompt 的缓存还没建立，成本天然偏高）

    # 策略 2：新 prompt 上线后前 24 小时标记为“预热期”
    # 预热期内不触发 Token 预算告警

    # 策略 3：如果 prompt 只是末尾加了规则（前缀没变）
    # → 利用 cache_control 的 ephemeral 标记新增部分
    # → 前缀部分继续命中缓存
    pass
```

### 5.3 最佳实践

| 做法 | 为什么 |
|---|---|
| 新规则尽量加在 prompt 末尾 | 前缀不变 → 旧缓存仍然有效 |
| 不要频繁改 prompt 开头的核心描述 | 开头变 = 全部缓存失效 |
| A/B 测试期间不按 token 成本判断好坏 | 新 prompt 的缓存还没热 |
| 大版本（major）升级后给 24h 预热期 | 让缓存重新建立 |

---

## 6、和其它章节的关系

| 章节 | 本章和它的关系 |
|---|---|
| 第 5 章 Cache Breakpoint | Prompt 变更直接影响 cache 命中率，需要 graceful migration |
| 第 10 章 prompts.yml | 版本管理取代了直接改 yml 文件 |
| 16-3 LangFuse | A/B 测试的 Rubric 分从 LangFuse Score 读取 |
| 16-6 灰度发布 | Prompt A/B 和模型灰度是同一套分流机制 |
| 18-1 进化全景 | 本章是 3×3 矩阵“Harness × Across Sessions/Users”的落地 |
| 18-2 数据飞轮 | P1 bad case 是 Auto-Prompt-Optimization 的输入 |

---

## 本章小结

到这里，Globex 的 Prompt 不再是“随手改的字符串”，而是一个有版本、有测试、能自动优化的系统：

1. **版本管理**：语义化版本号 + changelog + 回滚能力，每次改动可追溯。
2. **A/B 测试**：10% 分流 + Rubric 对比 + 3 天观察 + 自动放量/回滚标准。
3. **Auto-Prompt-Optimization**：LLM 分析 bad case → 生成修改建议 → 自动 A/B 验证 → 合格自动合入。
4. **安全门禁**：修改幅度 / 工具描述不可删 / fork 规则不可改 / 日均上限。
5. **Cache 友好**：新规则加末尾 / 大版本给预热期 / A/B 期不按 token 成本判断。
6. **从手动到全自动**：30% 的退化问题可以不训模型、分钟到天级自动修复。

下一章 **「记忆层自进化与成功策略沉淀」** 会讲进化闭环里另一条“轻路径”——让 Store 不只存偏好，还能存成功策略，让 Agent 的每一次成功经验都能被未来的自己复用。
