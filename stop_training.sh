# 查找所有相关进程
PROCESSES=$(ps aux | grep -E "(pretrain.py|accelerate|pt_elasti)" | grep -v grep)

if [ -z "$PROCESSES" ]; then
    echo "没有发现运行中的训练进程"
    exit 0
fi

echo "发现的进程："
echo "$PROCESSES" | while read line; do
    PID=$(echo $line | awk '{print $2}')
    CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    echo "  PID: $PID - $CMD"
done

echo ""
read -p "是否要关闭所有训练进程? (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消操作"
    exit 0
fi

echo "正在关闭训练进程..."

# 方法1: 关闭所有 pretrain.py 进程
echo "  关闭 pretrain.py 进程..."
pkill -9 -f "pretrain.py" 2>/dev/null

# 方法2: 关闭 accelerate launch 进程
echo "  关闭 accelerate launch 进程..."
pkill -9 -f "accelerate launch" 2>/dev/null

# 方法3: 关闭 pt_elastic 进程
echo "  关闭 pt_elastic 进程..."
pkill -9 -f "pt_elasti" 2>/dev/null

# 等待进程关闭
sleep 2

# 检查端口29500
echo ""
echo "检查端口29500..."
if lsof -i :29500 >/dev/null 2>&1; then
    echo "  端口29500仍被占用，正在强制关闭..."
    lsof -ti :29500 | xargs -r kill -9 2>/dev/null
    sleep 1
fi

# 最终验证
echo ""
REMAINING=$(ps aux | grep -E "(pretrain.py|accelerate|pt_elasti)" | grep -v grep)
if [ -z "$REMAINING" ]; then
    echo "所有训练进程已成功关闭"
    
    # 检查端口
    if ! lsof -i :29500 >/dev/null 2>&1; then
        echo "端口29500已释放"
    else
        echo "端口29500仍被占用，可能需要手动检查"
    fi
else
    echo "仍有进程在运行："
fi

echo ""
echo "当前GPU使用情况："
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  (无法获取GPU信息)"
