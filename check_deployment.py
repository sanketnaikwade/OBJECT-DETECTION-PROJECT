#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script to verify all deployment requirements are met
Run this before deploying to Render
"""

import os
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def check_file_exists(filepath, description):
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    if exists and filepath.endswith('.pt'):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"   📊 Size: {size_mb:.2f} MB")
    return exists

def check_requirements():
    required_packages = [
        'Flask',
        'flask-socketio',
        'eventlet',
        'Pillow',
        'ultralytics',
        'opencv-python-headless',
        'numpy',
        'torch',
        'torchvision',
        'gunicorn'
    ]
    
    print("\n📦 Checking requirements.txt...")
    with open('requirements.txt', 'r') as f:
        content = f.read().lower()
        
    for package in required_packages:
        if package.lower() in content:
            print(f"✅ {package}")
        else:
            print(f"❌ {package} - MISSING!")

def main():
    print("=" * 60)
    print("🔍 Render Deployment Checklist")
    print("=" * 60)
    
    print("\n📁 Checking required files...")
    all_ok = True
    all_ok &= check_file_exists('main.py', 'Main application')
    all_ok &= check_file_exists('requirements.txt', 'Requirements')
    all_ok &= check_file_exists('render.yaml', 'Render config')
    all_ok &= check_file_exists('runtime.txt', 'Python version')
    all_ok &= check_file_exists('yolov8n.pt', 'YOLO model')
    all_ok &= check_file_exists('templates/index.html', 'Frontend template')
    
    check_requirements()
    
    print("\n🔧 Configuration checks...")
    with open('main.py', 'r', encoding='utf-8') as f:
        main_content = f.read()
        if 'YOLO_CONFIG_DIR' in main_content:
            print("✅ YOLO cache directory configured")
        if 'PORT' in main_content and 'environ' in main_content:
            print("✅ Port configuration present")
    
    with open('render.yaml', 'r', encoding='utf-8') as f:
        render_content = f.read()
        if 'python main.py' in render_content:
            print("✅ Start command configured")
        if '/tmp' in render_content:
            print("✅ Temp directory configured for YOLO")
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed! Ready to deploy to Render")
        print("=" * 60)
        print("\n📤 Next steps:")
        print("1. Commit and push to GitHub:")
        print("   git add .")
        print("   git commit -m 'Fix deployment configuration'")
        print("   git push")
        print("\n2. Deploy on Render (auto-deploy or manual)")
        print("\n3. Check logs in Render dashboard")
    else:
        print("❌ Some checks failed. Fix the issues before deploying.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()
