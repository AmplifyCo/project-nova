#!/bin/bash
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Migrating from claude-agent to digital-twin"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get current user
CURRENT_USER=$(whoami)

# Determine current directory name and target
CURRENT_DIR=$(pwd)
CURRENT_DIR_NAME=$(basename "$CURRENT_DIR")

echo "📋 Current directory: $CURRENT_DIR"
echo "👤 Current user: $CURRENT_USER"
echo ""

# Step 1: Update git remote
echo "1️⃣  Updating git remote URL..."
if git remote get-url origin 2>/dev/null | grep -q "autonomous-claude-agent"; then
    git remote set-url origin https://github.com/AmplifyCo/digital-twin.git
    echo "   ✅ Git remote updated to: https://github.com/AmplifyCo/digital-twin.git"
else
    echo "   ℹ️  Git remote already updated or not using old URL"
fi
echo ""

# Step 2: Pull latest changes
echo "2️⃣  Pulling latest changes..."
git pull origin main
echo "   ✅ Repository updated"
echo ""

# Step 3: Stop old service
echo "3️⃣  Stopping old claude-agent service..."
if sudo systemctl is-active --quiet claude-agent 2>/dev/null; then
    sudo systemctl stop claude-agent
    echo "   ✅ Service stopped"
else
    echo "   ℹ️  Service not running or doesn't exist"
fi
echo ""

# Step 4: Disable and remove old service
echo "4️⃣  Removing old service files..."
if [ -f /etc/systemd/system/claude-agent.service ]; then
    sudo systemctl disable claude-agent 2>/dev/null || true
    sudo rm /etc/systemd/system/claude-agent.service
    echo "   ✅ Removed /etc/systemd/system/claude-agent.service"
else
    echo "   ℹ️  Old service file not found"
fi

if [ -f /etc/sudoers.d/claude-agent ]; then
    sudo rm /etc/sudoers.d/claude-agent
    echo "   ✅ Removed /etc/sudoers.d/claude-agent"
else
    echo "   ℹ️  Old sudoers file not found"
fi
echo ""

# Step 5: Create new service
echo "5️⃣  Creating new digital-twin service..."
sudo tee /etc/systemd/system/digital-twin.service > /dev/null << EOF
[Unit]
Description=Digital Twin - Self-Building AI System with Dual Brain Architecture
After=network.target

[Service]
Type=simple
User=$CURRENT_USER
WorkingDirectory=$CURRENT_DIR
Environment="PATH=$CURRENT_DIR/venv/bin"
Environment="PYTHONUNBUFFERED=1"

# Start agent in self-build mode
ExecStart=$CURRENT_DIR/venv/bin/python src/main.py

# Auto-restart on failure
Restart=always
RestartSec=10
StartLimitInterval=300
StartLimitBurst=5

# Logging
StandardOutput=append:$CURRENT_DIR/data/logs/agent.log
StandardError=append:$CURRENT_DIR/data/logs/error.log

# Resource limits
MemoryLimit=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF
echo "   ✅ Created /etc/systemd/system/digital-twin.service"
echo ""

# Step 6: Create new sudoers file
echo "6️⃣  Creating new sudoers configuration..."
sudo tee /etc/sudoers.d/digital-twin > /dev/null << 'SUDOERS_EOF'
# Limited sudo access for autonomous agent
# Package management
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/yum install *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/yum update *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/yum remove *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/apt-get install *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/apt-get update *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/apt install *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/apt update *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/pip install *

# Service management (only digital-twin service)
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart digital-twin
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/systemctl status *
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/systemctl stop digital-twin
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/systemctl start digital-twin

# Firewall management
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/firewall-cmd *

# Log viewing
ec2-user ALL=(ALL) NOPASSWD: /usr/bin/journalctl *
SUDOERS_EOF

sudo chmod 440 /etc/sudoers.d/digital-twin
echo "   ✅ Created /etc/sudoers.d/digital-twin"
echo ""

# Step 7: Reload systemd and enable new service
echo "7️⃣  Enabling new service..."
sudo systemctl daemon-reload
sudo systemctl enable digital-twin
echo "   ✅ Service enabled"
echo ""

# Step 8: Rename directory if needed
if [ "$CURRENT_DIR_NAME" = "autonomous-claude-agent" ]; then
    echo "8️⃣  Directory rename required..."
    echo "   Current: $CURRENT_DIR"
    echo "   Target:  $(dirname "$CURRENT_DIR")/digital-twin"
    echo ""
    echo "   ⚠️  This script cannot rename the directory it's running from."
    echo "   After this script completes, run:"
    echo ""
    echo "   cd ~"
    echo "   mv autonomous-claude-agent digital-twin"
    echo "   cd digital-twin"
    echo "   sudo systemctl start digital-twin"
    echo ""
    NEEDS_DIR_RENAME=true
else
    echo "8️⃣  Starting new service..."
    sudo systemctl start digital-twin
    echo "   ✅ Service started"
    echo ""
    NEEDS_DIR_RENAME=false
fi

# Step 9: Verify status
echo "9️⃣  Verifying installation..."
if sudo systemctl is-active --quiet digital-twin; then
    echo "   ✅ Service is running!"
    sudo systemctl status digital-twin --no-pager | head -10
else
    if [ "$NEEDS_DIR_RENAME" = true ]; then
        echo "   ⏸️  Service will start after directory rename"
    else
        echo "   ⚠️  Service is not running. Check logs:"
        echo "      sudo journalctl -u digital-twin -n 50"
    fi
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Migration Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "New commands:"
echo "  • Start:   sudo systemctl start digital-twin"
echo "  • Stop:    sudo systemctl stop digital-twin"
echo "  • Restart: sudo systemctl restart digital-twin"
echo "  • Status:  sudo systemctl status digital-twin"
echo "  • Logs:    sudo journalctl -u digital-twin -f"
echo ""

if [ "$NEEDS_DIR_RENAME" = true ]; then
    echo "⚠️  IMPORTANT: Complete the migration by renaming the directory:"
    echo ""
    echo "   cd ~"
    echo "   mv autonomous-claude-agent digital-twin"
    echo "   cd digital-twin"
    echo "   sudo systemctl start digital-twin"
    echo ""
fi

echo "To remove this migration script:"
echo "  rm migrate-to-digital-twin.sh"
echo ""
