DJ_PATH="/mnt/data/workspace/sy_transformers/deepseekv3/v3/data/data-juicer"

CONFIG_FILES=(
    "minidsv3_text_skypile.yaml"
)

for CONFIG in "${CONFIG_FILES[@]}"; do
    echo "Processing $CONFIG..."
    nohup python $DJ_PATH/tools/process_data.py --config "$CONFIG" | tee "dj_log_${CONFIG%.*}.log" &
    sleep 5  
done

echo "All processes started! Check logs: dj_log_*"

