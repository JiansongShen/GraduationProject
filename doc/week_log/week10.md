# 第10周周记

**时间**：第10周  
**主题**：FastAPI 服务与上传流程

本周学习用 FastAPI 构建 REST 接口：健康检查、文件上传、异步任务与错误码约定。阅读 `ui/main.py` 中路由与配置加载逻辑，理解 `Config`、`RiskConfig` 如何驱动模型路径与设备选择。

完成工作：本地启动服务，用 curl 或前端页面上传小体积测试体；打通 `prepare_uploaded_volume_for_dataset` 与 `run_patch_based_segmentation` 调用链；为上传目录设置清理策略，避免磁盘占满。

**收获与问题**：CORS 与跨域在联调阶段常踩坑，需与前端约定同源或白名单；大文件上传应限制类型与大小并记录日志。下周完善风险预测接口与返回 JSON 结构。
