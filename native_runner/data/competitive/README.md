# Environment data and compatibility · 环境数据与兼容性

[English home](../../../README.md) · [中文首页](../../../README.zh-CN.md) · [Architecture / 架构](../../../docs/architecture.md)

These versioned tables connect native battle mechanics to policy features and action decoding. They define the supported cards, forms, and catalog identities used by the V4 model.

本目录的版本化数据将原生对局机制连接到策略特征与动作解码，定义 V4 模型使用的卡牌、形态和目录标识。

## File map · 文件说明

| File / 文件 | Purpose / 用途 |
| --- | --- |
| `card_specs.json.gz` | Card attributes and feature inputs / 卡牌属性与特征输入 |
| `card_logic.json.gz` | Card behavior and mechanic rules / 卡牌行为与机制规则 |
| `effects.json.gz`, `projectiles.json.gz` | Effect and projectile semantics / 效果与弹射物语义 |
| `model_catalogs.json.gz` | Model vocabulary and catalog ordering / 模型词表与目录顺序 |
| [runtime.json](runtime.json) | Battle timelines and supported content identities / 对局时间线与支持资源标识 |
| [card_support.json](card_support.json) | 122 supported cards / 122 张支持卡牌 |
| [form_evidence.json](form_evidence.json) | 62 form and ability compatibility bindings / 62 项形态与技能兼容绑定 |
| [manifest.json](manifest.json) | SHA-256 of decompressed JSON and compiler input provenance / 解压后 JSON 的 SHA-256 与编译输入标识 |

## Version maintenance · 版本维护

To rebuild the five compiled tables, provide matching compiler inputs locally:

重新生成五张编译表时，在本地提供匹配的编译输入：

```sh
python -m native_runner.resource_compiler --workspace /path/to/resources --output /path/to/local-data
```

Set `CR_COMPETITIVE_DATA_ROOT` to test an alternate compiled dataset. Catalog order and feature semantics must remain compatible with the checkpoint; engine, probe, and content identities are also checked on connection. A version update requires encoding and native-execution parity checks alongside regenerated tables.

测试另一份编译数据时设置 `CR_COMPETITIVE_DATA_ROOT`。目录顺序及特征语义须与 checkpoint 兼容，连接时还会核对引擎、probe 与资源标识。更新版本时，除重新生成数据表外，还需验证编码与原生执行的一致性。
