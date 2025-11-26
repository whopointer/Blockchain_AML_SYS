#!/usr/bin/env python3
"""
Elliptic数据集下载脚本
自动下载并解压Elliptic区块链数据集
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
import argparse

def download_kaggle_dataset(data_dir: str = "data/raw"):
    """
    从Kaggle下载Elliptic数据集
    需要先配置Kaggle API密钥
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ 请先安装kaggle包: pip install kaggle")
        return False
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        # 初始化Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        print("📥 开始下载Elliptic数据集...")
        
        # 下载数据集
        api.dataset_download_files(
            'ellipticco/elliptic-data-set', 
            path=data_dir,
            unzip=True
        )
        
        print("✅ 数据集下载完成！")
        print(f"📁 数据保存在: {os.path.abspath(data_dir)}")
        
        # 列出下载的文件
        downloaded_files = os.listdir(data_dir)
        print("\n📋 下载的文件:")
        for file in sorted(downloaded_files):
            file_path = os.path.join(data_dir, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("\n💡 请确保:")
        print("1. 已安装kaggle包: pip install kaggle")
        print("2. 已配置Kaggle API密钥:")
        print("   - 在 ~/.kaggle/kaggle.json 中配置")
        print("   - 或设置环境变量 KAGGLE_USERNAME 和 KAGGLE_KEY")
        return False

def download_direct_links(data_dir: str = "data/raw"):
    """
    使用直接链接下载（如果可用）
    """
    print("📥 尝试使用直接链接下载...")
    
    # 创建数据目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 注意：这些链接可能需要根据实际情况更新
    files_to_download = {
        "elliptic_txs_classes.csv": "直接链接1",
        "elliptic_txs_edgelist.csv": "直接链接2", 
        "elliptic_txs_features.csv": "直接链接3"
    }
    
    success_count = 0
    
    for filename, url in files_to_download.items():
        if url == "直接链接":
            print(f"⚠️  {filename}: 需要手动下载")
            continue
            
        try:
            print(f"📥 下载 {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = os.path.join(data_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ {filename} 下载完成")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {filename} 下载失败: {str(e)}")
    
    if success_count > 0:
        print(f"\n✅ 成功下载 {success_count} 个文件")
        return True
    else:
        print("\n❌ 没有文件下载成功，请使用Kaggle方式或手动下载")
        return False

def verify_data_files(data_dir: str = "data/raw"):
    """
    验证数据文件是否存在和完整
    """
    required_files = [
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv", 
        "elliptic_txs_features.csv"
    ]
    
    missing_files = []
    existing_files = []
    
    for filename in required_files:
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            existing_files.append((filename, size_mb))
        else:
            missing_files.append(filename)
    
    print("\n📊 数据文件验证结果:")
    
    if existing_files:
        print("✅ 已存在的文件:")
        for filename, size_mb in existing_files:
            print(f"  - {filename} ({size_mb:.1f} MB)")
    
    if missing_files:
        print("❌ 缺失的文件:")
        for filename in missing_files:
            print(f"  - {filename}")
    
    return len(missing_files) == 0

def main():
    parser = argparse.ArgumentParser(description='下载Elliptic数据集')
    parser.add_argument('--method', choices=['kaggle', 'direct', 'verify'], 
                       default='kaggle', help='下载方法')
    parser.add_argument('--data-dir', default='data/raw', 
                       help='数据保存目录')
    
    args = parser.parse_args()
    
    print("🚀 Elliptic数据集下载工具")
    print("=" * 50)
    
    if args.method == 'verify':
        success = verify_data_files(args.data_dir)
        if success:
            print("\n✅ 所有数据文件都存在！")
        else:
            print("\n❌ 部分数据文件缺失，请运行下载命令")
    
    elif args.method == 'kaggle':
        success = download_kaggle_dataset(args.data_dir)
        if success:
            verify_data_files(args.data_dir)
    
    elif args.method == 'direct':
        success = download_direct_links(args.data_dir)
        if success:
            verify_data_files(args.data_dir)

if __name__ == "__main__":
    main()