#!/bin/bash

# ================= 配置区域 =================

# 1. 数据集列表
#DATASETS=("HAR_inertial" "DailySports" "UWave" "WISDM" "GRABMyo")
DATASETS=("GRABMyo")
# 2. 模型列表 (消融实验的三个关键模型)
#MODELS=("ClassifierBaseline" "ClassifierHSIC" "ClassifierDualPathNoIB")
MODELS=("ClassifierHSIC")
# 3. 骨干网络列表
NETWORKS=("cnn")
#NETWORKS=("transformer")

# 4. 种子范围 (1975 - 2025)
# Shell 中 {start..end} 会自动展开
SEEDS=({1975..2025})

# ================= 执行区域 =================

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        for net in "${NETWORKS[@]}"; do
            for seed in "${SEEDS[@]}"; do

                echo "======================================================================="
                echo "[运行中] Data: $dataset | Model: $model | Net: $net | Seed: $seed"
                echo "======================================================================="

                # 执行 Python 命令
                # 建议加上 > /dev/null 来减少屏幕输出，或者重定向到日志文件
                python main_ablation.py \
                    --dataset "$dataset" \
                    --model "$model" \
                    --network "$net" \
                    --seed "$seed"\
                    --test



                # 检查上一步是否报错，如果有错则停止 (可选)
                if [ $? -ne 0 ]; then
                    echo "Error occurred at Seed $seed! Stopping..."
                    exit 1
                fi

            done
        done
    done
done

echo "所有消融实验任务已完成。"