#!/bin/bash
# System optimization script for Linux CUDA systems

echo "========================================"
echo "Linux System Optimization for AlphaZero"
echo "========================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "Applying system optimizations..."

# 1. CPU Performance Settings
echo ""
echo "1. Setting CPU governor to performance..."
if command -v cpupower &> /dev/null; then
    cpupower frequency-set -g performance
    echo -e "${GREEN}✓ CPU governor set to performance${NC}"
else
    # Fallback method
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance > $cpu 2>/dev/null
    done
    echo -e "${GREEN}✓ CPU governor set to performance (manual)${NC}"
fi

# 2. Disable CPU frequency scaling
echo ""
echo "2. Disabling CPU frequency scaling..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/; do
    if [ -f "$cpu/scaling_max_freq" ] && [ -f "$cpu/cpuinfo_max_freq" ]; then
        cat "$cpu/cpuinfo_max_freq" > "$cpu/scaling_max_freq"
        cat "$cpu/cpuinfo_max_freq" > "$cpu/scaling_min_freq"
    fi
done
echo -e "${GREEN}✓ CPU frequency scaling disabled${NC}"

# 3. GPU Optimization
echo ""
echo "3. Optimizing GPU settings..."
if command -v nvidia-smi &> /dev/null; then
    # Enable persistence mode
    nvidia-smi -pm 1
    echo -e "${GREEN}✓ GPU persistence mode enabled${NC}"
    
    # Set GPU to maximum performance
    nvidia-smi -i 0 -ac 5001,1695 2>/dev/null || echo -e "${YELLOW}! Could not set GPU clocks (may require specific GPU)${NC}"
    
    # Set compute mode to exclusive
    nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
    echo -e "${GREEN}✓ GPU compute mode set to exclusive${NC}"
else
    echo -e "${YELLOW}! nvidia-smi not found, skipping GPU optimization${NC}"
fi

# 4. Memory Settings
echo ""
echo "4. Optimizing memory settings..."

# Increase shared memory
sysctl -w kernel.shmmax=68719476736
sysctl -w kernel.shmall=4294967296

# Optimize swappiness
sysctl -w vm.swappiness=10

# Enable transparent huge pages
echo always > /sys/kernel/mm/transparent_hugepage/enabled
echo always > /sys/kernel/mm/transparent_hugepage/defrag

echo -e "${GREEN}✓ Memory settings optimized${NC}"

# 5. I/O Scheduler
echo ""
echo "5. Setting I/O scheduler..."
for disk in /sys/block/sd*/queue/scheduler; do
    echo noop > $disk 2>/dev/null || echo deadline > $disk 2>/dev/null
done
echo -e "${GREEN}✓ I/O scheduler optimized${NC}"

# 6. Network Optimization (for distributed training)
echo ""
echo "6. Optimizing network settings..."
sysctl -w net.core.rmem_max=134217728
sysctl -w net.core.wmem_max=134217728
sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
echo -e "${GREEN}✓ Network settings optimized${NC}"

# 7. Process Limits
echo ""
echo "7. Setting process limits..."
cat >> /etc/security/limits.conf <<EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
* soft memlock unlimited
* hard memlock unlimited
EOF
echo -e "${GREEN}✓ Process limits updated${NC}"

# 8. Create optimized run script
echo ""
echo "8. Creating optimized run script..."
cat > /usr/local/bin/run_alphazero_optimized <<'EOF'
#!/bin/bash
# Optimized launcher for AlphaZero

# Set environment
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export OMP_PROC_BIND=TRUE
export OMP_PLACES=cores
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Set process priority
renice -n -10 $$

# Set CPU affinity (use first 16 cores)
taskset -c 0-15 $@
EOF

chmod +x /usr/local/bin/run_alphazero_optimized
echo -e "${GREEN}✓ Optimized run script created${NC}"

# 9. Save settings
echo ""
echo "9. Making settings persistent..."

# Save sysctl settings
cat > /etc/sysctl.d/99-alphazero.conf <<EOF
# AlphaZero optimizations
kernel.shmmax = 68719476736
kernel.shmall = 4294967296
vm.swappiness = 10
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
EOF

# Create systemd service for persistence
cat > /etc/systemd/system/alphazero-optimize.service <<EOF
[Unit]
Description=AlphaZero System Optimization
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/opt/pytorch-alpha-zero/optimize_linux_system.sh
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo -e "${GREEN}✓ Settings saved${NC}"

# Summary
echo ""
echo "========================================"
echo "Optimization Complete!"
echo "========================================"
echo ""
echo "System optimizations applied:"
echo "  ✓ CPU governor: performance"
echo "  ✓ CPU frequency: locked to maximum"
echo "  ✓ GPU: persistence mode, exclusive compute"
echo "  ✓ Memory: huge pages enabled, swappiness reduced"
echo "  ✓ I/O: optimized scheduler"
echo "  ✓ Network: increased buffers"
echo "  ✓ Process limits: increased"
echo ""
echo "To run AlphaZero with optimizations:"
echo "  run_alphazero_optimized python3 playchess_cuda.py --model model.pt"
echo ""
echo "To make optimizations permanent:"
echo "  systemctl enable alphazero-optimize.service"
echo ""
echo "Note: Reboot recommended for all settings to take effect."