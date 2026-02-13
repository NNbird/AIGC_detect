// 这是一个极简的 Node 服务器，专门用来发布 React 静态文件
const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');

const app = express();
const PORT = parseInt(process.env.PORT) || 3000; // 默认端口，宝塔会覆盖它

// 1. 配置反向代理 (转发 API 请求到 Python 后端)
// 这样前端请求 /api/predict 就会被转发到 localhost:5000
app.use(
  '/api',
  createProxyMiddleware({
    target: 'http://127.0.0.1:5000', // Python 后端地址
    changeOrigin: true,
    pathRewrite: {
      '^/api': '/api', // 保持路径不变，或者根据 Flask 路由调整
    },
  })
);

// 2. 托管静态文件 (dist 目录)
app.use(express.static(path.join(__dirname, 'dist')));

// 3. 处理前端路由 (所有其他请求都返回 index.html)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

app.listen(PORT, () => {
  console.log(`前端服务器已启动: http://localhost:${PORT}`);
});
