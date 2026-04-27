# Gradulate Frontend (Vue3)

这是 Gradulate 项目的前端子项目，使用 Vue3 + Vite，实现 CTA 上传、分割展示、风险预测交互界面。

## 目录

- `src/App.vue`: 主要业务界面
- `src/style.css`: 页面样式
- `vite.config.js`: 本地开发代理配置

## 启动方式

先启动后端 API（默认 `http://127.0.0.1:8000`）：

```bash
python script/start_web.py
```

然后启动前端：

```bash
cd frontend
npm install
npm run dev
```

默认访问地址：`http://127.0.0.1:5173`

## 后端地址配置

- 本地开发默认使用 Vite 代理（`/api` 与 `/files`）
- 如果前端与后端分开部署，可设置环境变量：

```bash
VITE_API_BASE=http://your-backend-host:8000
```
