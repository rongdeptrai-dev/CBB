#!/bin/bash

# ==========================================
# CyberBot AI Deployment Script for AWS G4dn.xlarge
# T4 GPU + 16GB VRAM + 16GB RAM + 4 vCPUs
# ==========================================

set -e

echo "🚀 Starting CyberBot AI deployment on AWS G4dn.xlarge..."
echo "🎮 Instance: T4 GPU (16GB VRAM) + 16GB RAM + 4 vCPUs"

# ========== System Information ==========
echo "📊 System Information:"
echo "- CPU: $(nproc) cores"
echo "- RAM: $(free -h | awk '/^Mem:/ {print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
    echo "- VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) MB"
else
    echo "- GPU: Not detected or drivers not installed"
fi
echo ""

# ========== Update System ==========
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ========== Install NVIDIA Drivers & CUDA (if needed) ==========
if ! command -v nvidia-smi &> /dev/null; then
    echo "🎮 Installing NVIDIA drivers and CUDA..."
    
    # Install NVIDIA drivers
    sudo apt install -y nvidia-driver-470
    
    # Install CUDA toolkit
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get -y install cuda
    
    # Add CUDA to PATH
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    source ~/.bashrc
    
    echo "⚠️ NVIDIA drivers installed. System reboot may be required."
    echo "Run 'sudo reboot' and re-run this script after reboot."
fi

# ========== Install Python & Dependencies ==========
echo "🐍 Setting up Python environment..."
sudo apt install -y python3.9 python3.9-pip python3.9-venv python3.9-dev

# Create virtual environment
python3.9 -m venv cyberbot_env
source cyberbot_env/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# ========== Install PyTorch with CUDA Support ==========
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA installation
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# ========== Install AI Dependencies ==========
echo "🤖 Installing AI and ML dependencies..."
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install bitsandbytes==0.41.3
pip install sentence-transformers==2.2.2
pip install datasets==2.14.0
pip install tokenizers==0.15.0

# Install specific model dependencies
pip install sentencepiece==0.1.99
pip install protobuf==4.25.0

# ========== Install Application Dependencies ==========
echo "📚 Installing application dependencies..."
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install sqlalchemy==2.0.23
pip install python-dotenv==1.0.0
pip install requests==2.31.0
pip install psutil==5.9.6
pip install numpy==1.24.3
pip install pandas==2.1.3

# Vector DB and Graph DB
pip install pinecone-client==2.2.4
pip install neo4j==5.13.0

# Additional utilities
pip install python-multipart==0.0.6
pip install jinja2==3.1.2
pip install aiofiles==23.2.1

# ========== Setup CyberBot Application ==========
echo "🔧 Setting up CyberBot application..."

# Create application directory
mkdir -p /opt/cyberbot
cd /opt/cyberbot

# Copy application files (assuming they're in current directory)
if [ -f "models/cyberbot_real.py" ]; then
    cp -r models/ /opt/cyberbot/
    cp requirements.txt /opt/cyberbot/ 2>/dev/null || echo "requirements.txt not found, skipping..."
    cp .env.g4dn-xlarge /opt/cyberbot/.env
else
    echo "⚠️ Application files not found in current directory"
    echo "Please copy your CyberBot files to /opt/cyberbot/"
fi

# Set permissions
sudo chown -R $USER:$USER /opt/cyberbot
chmod +x /opt/cyberbot/models/cyberbot_real.py

# ========== Create Systemd Service ==========
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/cyberbot.service > /dev/null <<EOF
[Unit]
Description=CyberBot AI Customer Service
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/cyberbot
Environment=PATH=/home/$USER/cyberbot_env/bin
ExecStart=/home/$USER/cyberbot_env/bin/python -m uvicorn models.cyberbot_real:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

# Resource limits for G4dn.xlarge
MemoryMax=14G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
EOF

# ========== Configure Firewall ==========
echo "🔒 Configuring firewall..."
sudo ufw allow 8000/tcp
sudo ufw allow ssh
sudo ufw --force enable

# ========== GPU Performance Optimization ==========
echo "⚡ Optimizing GPU performance..."

# Set GPU performance mode
sudo nvidia-smi -pm 1

# Set maximum GPU clocks
sudo nvidia-smi -ac 5001,1590

# Enable persistence mode
sudo nvidia-smi -i 0 -pm 1

# ========== Create Monitoring Scripts ==========
echo "📊 Creating monitoring scripts..."

# GPU monitoring script
tee /opt/cyberbot/monitor_gpu.sh > /dev/null <<EOF
#!/bin/bash
while true; do
    echo "=== GPU Status \$(date) ==="
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv
    echo ""
    sleep 10
done
EOF

chmod +x /opt/cyberbot/monitor_gpu.sh

# System monitoring script
tee /opt/cyberbot/monitor_system.sh > /dev/null <<EOF
#!/bin/bash
while true; do
    echo "=== System Status \$(date) ==="
    echo "CPU Usage: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | awk -F'%' '{print \$1}')"
    echo "RAM Usage: \$(free | grep Mem | awk '{printf "%.1f%%", \$3/\$2 * 100.0}')"
    echo "Disk Usage: \$(df / | awk 'NR==2{printf "%.1f%%", \$5}')"
    echo ""
    sleep 30
done
EOF

chmod +x /opt/cyberbot/monitor_system.sh

# ========== Test GPU and PyTorch ==========
echo "🧪 Testing GPU and PyTorch setup..."
python3 -c "
import torch
import transformers
print(f'PyTorch version: {torch.__version__}')
print(f'Transformers version: {transformers.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
    print('GPU test passed! ✅')
else:
    print('GPU test failed! ❌')
"

# ========== Pre-download Models ==========
echo "📥 Pre-downloading AI models..."
python3 -c "
from transformers import AutoTokenizer, T5ForConditionalGeneration
import torch

try:
    print('Downloading VietAI/vit5-large...')
    tokenizer = AutoTokenizer.from_pretrained('VietAI/vit5-large')
    model = T5ForConditionalGeneration.from_pretrained('VietAI/vit5-large', torch_dtype=torch.float16)
    print('✅ VietAI/vit5-large downloaded successfully')
    
    print('Downloading emotion model...')
    from transformers import pipeline
    emotion_pipeline = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
    print('✅ Emotion model downloaded successfully')
    
    print('Downloading sentiment model...')
    sentiment_pipeline = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest')
    print('✅ Sentiment model downloaded successfully')
    
except Exception as e:
    print(f'⚠️ Model download error: {e}')
    print('Models will be downloaded on first use')
"

# ========== Start Services ==========
echo "🚀 Starting CyberBot services..."
sudo systemctl daemon-reload
sudo systemctl enable cyberbot.service
sudo systemctl start cyberbot.service

# Wait for service to start
sleep 5

# Check service status
echo "📊 Service status:"
sudo systemctl status cyberbot.service --no-pager

# ========== Final Instructions ==========
echo ""
echo "🎉 CyberBot AI deployment completed!"
echo ""
echo "📋 Service Management:"
echo "  Start:   sudo systemctl start cyberbot"
echo "  Stop:    sudo systemctl stop cyberbot"
echo "  Restart: sudo systemctl restart cyberbot"
echo "  Status:  sudo systemctl status cyberbot"
echo "  Logs:    journalctl -u cyberbot -f"
echo ""
echo "📊 Monitoring:"
echo "  GPU:     ./monitor_gpu.sh"
echo "  System:  ./monitor_system.sh"
echo "  API:     curl http://localhost:8000/health"
echo ""
echo "🌐 API Endpoints:"
echo "  Health:  http://localhost:8000/health"
echo "  Chat:    http://localhost:8000/chat"
echo "  Models:  http://localhost:8000/models/status"
echo ""
echo "⚡ Performance Tips:"
echo "  - Monitor GPU usage with: watch nvidia-smi"
echo "  - Check memory usage with: free -h"
echo "  - Optimize batch size in .env file if needed"
echo ""
echo "🔧 Configuration: /opt/cyberbot/.env"
echo "📁 Application: /opt/cyberbot/"
echo ""

# Test API
echo "🧪 Testing API endpoint..."
sleep 10
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding successfully!"
else
    echo "❌ API test failed. Check logs: journalctl -u cyberbot -f"
fi

echo ""
echo "✨ Deployment complete! Your AI chatbot is ready on G4dn.xlarge! ✨"
