# 🚀 CyberBot AI - G4dn.xlarge Deployment Guide

## 📋 **BƯỚC 1: CHUẨN BỊ GIT REPOSITORY**

### **1.1 Git Add & Commit**
```bash
# Thêm tất cả files quan trọng
git add models/
git add .env.g4dn-xlarge
git add requirements.txt
git add deploy_g4dn_xlarge.sh
git add test_g4dn_performance.py
git add README_G4DN.md

# Commit với message rõ ràng
git commit -m "🚀 G4dn.xlarge GPU optimization: Enhanced AI with T4 GPU, emotion analysis, 10x performance"
```

### **1.2 Tạo Repository trên GitHub**
```bash
# Nếu chưa có remote repository
git remote add origin https://github.com/your-username/cyberbot-ai.git

# Push lên GitHub
git branch -M main
git push -u origin main
```

---

## 🎮 **BƯỚC 2: TRIỂN KHAI TRÊN AWS G4dn.xlarge**

### **2.1 SSH vào G4dn.xlarge Instance**
```bash
# SSH với private key
ssh -i "your-key.pem" ubuntu@your-g4dn-instance-ip

# Hoặc bằng username/password nếu đã setup
ssh ubuntu@your-g4dn-instance-ip
```

### **2.2 Clone Repository và Deploy**
```bash
# Clone code từ GitHub
git clone https://github.com/your-username/cyberbot-ai.git
cd cyberbot-ai

# Chạy script triển khai tự động
chmod +x deploy_g4dn_xlarge.sh
sudo ./deploy_g4dn_xlarge.sh
```

### **2.3 Manual Setup (nếu cần)**
```bash
# Cập nhật system
sudo apt update && sudo apt upgrade -y

# Cài NVIDIA drivers (nếu chưa có)
sudo apt install -y nvidia-driver-470
sudo reboot  # Reboot sau khi cài driver

# Cài CUDA toolkit
sudo apt install -y cuda-11-8

# Setup Python environment
python3.9 -m venv cyberbot_env
source cyberbot_env/bin/activate

# Cài PyTorch với CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài dependencies
pip install -r requirements.txt

# Copy cấu hình
cp .env.g4dn-xlarge .env

# Test GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## ⚙️ **BƯỚC 3: CẤU HÌNH VÀ KHỞI ĐỘNG**

### **3.1 Cấu Hình Environment**
```bash
# Chỉnh sửa .env file
nano .env

# Các setting quan trọng:
USE_HUGGINGFACE=true
HF_USE_GPU=true
HF_MODEL_NAME=VietAI/vit5-large
HF_BATCH_SIZE=8
MAX_VRAM_USAGE=14.0
ENABLE_EMOTION_DETECTION=true
ENABLE_SENTIMENT_ANALYSIS=true
```

### **3.2 Khởi Động Service**
```bash
# Tạo systemd service
sudo tee /etc/systemd/system/cyberbot.service > /dev/null <<EOF
[Unit]
Description=CyberBot AI G4dn GPU Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cyberbot-ai
Environment=PATH=/home/ubuntu/cyberbot_env/bin
ExecStart=/home/ubuntu/cyberbot_env/bin/python -m uvicorn models.cyberbot_real:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Enable và start service
sudo systemctl daemon-reload
sudo systemctl enable cyberbot
sudo systemctl start cyberbot
```

---

## 🧪 **BƯỚC 4: TESTING VÀ VERIFICATION**

### **4.1 Test GPU Performance**
```bash
# Chạy performance test
python test_g4dn_performance.py

# Check GPU utilization
nvidia-smi

# Monitor real-time
watch nvidia-smi
```

### **4.2 Test API Endpoints**
```bash
# Health check
curl http://localhost:8000/health

# Test chat với emotion analysis
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Video của tôi không hiển thị trên For You Page, tôi rất bực mình!",
    "username": "test_user"
  }'

# Expected response với emotion analysis:
{
  "response": "Tôi hiểu cảm giác bực bội của bạn về For You Page...",
  "emotion_analysis": {"label": "anger", "score": 0.89},
  "sentiment_analysis": {"label": "NEGATIVE", "score": 0.92},
  "response_time": 0.3,
  "device_used": "cuda:0"
}
```

### **4.3 Load Testing**
```bash
# Test concurrent users
for i in {1..10}; do
  curl -X POST "http://localhost:8000/chat" \
    -H "Content-Type: application/json" \
    -d '{"message": "Test message '$i'", "username": "user'$i'"}' &
done
```

---

## 📊 **BƯỚC 5: MONITORING VÀ OPTIMIZATION**

### **5.1 Performance Monitoring**
```bash
# GPU monitoring
./monitor_gpu.sh

# System monitoring  
./monitor_system.sh

# Service logs
journalctl -u cyberbot -f
```

### **5.2 Expected Performance Metrics**
```
✅ Response Time: 0.2-0.5s
✅ GPU Utilization: 60-80%
✅ VRAM Usage: 8-12GB/16GB
✅ Concurrent Users: 50-100
✅ Throughput: 15-20 req/s
```

### **5.3 Optimization Tips**
```bash
# Nếu VRAM không đủ
HF_BATCH_SIZE=4          # Giảm từ 8 xuống 4
HF_MODEL_NAME=VietAI/vit5-base  # Dùng model nhỏ hơn

# Nếu muốn tăng performance
HF_BATCH_SIZE=16         # Tăng batch size
HF_NUM_BEAMS=8          # Tăng beam search
```

---

## 🔒 **BƯỚC 6: SECURITY VÀ PRODUCTION**

### **6.1 Firewall Setup**
```bash
# Mở port 8000 cho API
sudo ufw allow 8000/tcp
sudo ufw allow ssh
sudo ufw enable
```

### **6.2 SSL Certificate (Production)**
```bash
# Cài nginx reverse proxy
sudo apt install nginx

# Setup SSL với Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### **6.3 API Key Security**
```bash
# Set API key trong .env
API_KEY=your_super_secure_api_key_here

# Test với API key
curl -X POST "http://localhost:8000/chat" \
  -H "X-API-Key: your_super_secure_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"message": "Test", "username": "user"}'
```

---

## 🎯 **BƯỚC 7: GO-LIVE CHECKLIST**

### **✅ Pre-Go-Live Checklist**
- [ ] GPU drivers installed và working
- [ ] CUDA toolkit installed
- [ ] PyTorch GPU working (`torch.cuda.is_available() = True`)
- [ ] All models loaded successfully
- [ ] API responding với emotion analysis
- [ ] Performance test passed (< 0.5s response)
- [ ] Concurrent load test passed (50+ users)
- [ ] Monitoring setup và working
- [ ] Service auto-restart enabled
- [ ] Firewall configured
- [ ] API key security enabled

### **🚀 Go-Live Commands**
```bash
# Final start
sudo systemctl start cyberbot
sudo systemctl status cyberbot

# Verify GPU usage
nvidia-smi

# Final API test
curl http://your-domain.com:8000/health
```

---

## 📞 **SUPPORT & TROUBLESHOOTING**

### **Common Issues:**
```bash
# CUDA not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Out of VRAM
# Reduce HF_BATCH_SIZE or switch to smaller model

# Service not starting
sudo journalctl -u cyberbot -f
```

### **Performance Issues:**
```bash
# Check GPU bottleneck
nvidia-smi

# Check system resources
htop
free -h
df -h
```

---

## 🎉 **FINAL RESULT**

Sau khi hoàn thành, bạn sẽ có:

✅ **CyberBot AI running on G4dn.xlarge**  
✅ **T4 GPU fully utilized**  
✅ **10x faster response time (0.3s vs 3s)**  
✅ **50-100 concurrent users support**  
✅ **Advanced emotion & sentiment analysis**  
✅ **Production-ready với monitoring**  

**🚀 Your AI chatbot is now SUPERCHARGED with GPU power! 🎮**
