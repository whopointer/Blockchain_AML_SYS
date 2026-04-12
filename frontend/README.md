# Blockchain AML System - Frontend

区块链反洗钱系统前端，基于 React + TypeScript 构建。

## 环境配置

### Node.js 版本要求

- **Node.js**: 建议使用版本 >= 22.x

建议使用 nvm (Node Version Manager) 管理 Node.js 版本：

```bash
# 安装 nvm (如未安装)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# 使用 nvm 安装 Node.js
nvm install 22
nvm use 22
```

### 项目依赖

首次 clone 项目后，需要安装依赖：

```bash
npm install
```

## 启动命令

| 命令            | 说明                                        |
| --------------- | ------------------------------------------- |
| `npm start`     | 启动开发服务器 (默认 http://localhost:3000) |
| `npm run build` | 构建生产环境版本 (输出到 `build/` 目录)     |

### 生产构建

```bash
npm run build
```

构建产物将生成在 `build/` 目录下，可直接部署到静态服务器 (Nginx 等)。

## 部署说明

### 生产环境部署

本项目已配置自动化部署到生产服务器：

| 配置项     | 值                      |
| ---------- | ----------------------- |
| 部署位置   | `/var/www/html`         |
| 监听端口   | 3000                    |
| Nginx 配置 | `/etc/nginx/nginx.conf` |

### 部署步骤

```bash
# 1. 构建生产版本
npm run build

# 2. 将构建产物上传到服务器
scp -r build/* root@210.28.133.13:/var/www/html/

# 3. 重启 Nginx 使配置生效
ssh root@210.28.133.13 "sudo systemctl reload nginx"
```

