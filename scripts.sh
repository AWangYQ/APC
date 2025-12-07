#!/bin/bash

# 设置检测间隔时间（秒）
interval=60
# 设置显存占用的阈值（单位：MB），小于此值则认为空闲
memory_threshold=500  # 例如500MB

while true; do
    # 获取GPU 0的显存使用情况
    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)

    # 判断GPU 0的显存是否低于阈值
    if [ "$memory_used" -lt "$memory_threshold" ]; then
        echo "GPU 0显存未被大量占用，开始运行代码..."
        # 在这里添加你想要运行的代码，例如：
        python train.py --config_file configs/MSMT17/ClipBase_olp.yml
        break  # 运行完代码后退出循环
    else
        echo "GPU 1显存占用较高，等待 $interval 秒后再次检测..."
    fi

    # 等待指定的检测间隔时间
    sleep "$interval"
done
