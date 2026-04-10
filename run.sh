#!/bin/bash

# ================= 配置区域 =================

# 1. 指定要使用的 GPU
DEVICE="cuda:2"

# 2. 指定要运行的数据集列表 (空格分隔)
# 你的代码支持: WISDM, GRABMyo, DailySports, UWave, HAR_inertial
DATASETS=("DailySports")
#DATASETS=("HAR_inertial" "DailySports" "UWave" "GRABMyo" "WISDM")

# 3. 指定要运行的模型列表 (空格分隔)
# 可选: NDCC, TLN, OpenMax, UNODE, LINe, OLED, NCI, KNFST, Local_KNFST
#MODELS=("NDCC" "TLN" "OpenMax" "UNODE" "LINe" "OLED" "NCI" "KNFST" "Local_KNFST" "KNFST_pre" "Local_KNFST_pre")
#MODELS=("KNFST" "Local_KNFST" "KNFST_pre" "Local_KNFST_pre")
MODELS=("NCI")
# ==========================================

# 第一层循环：遍历数据集
for dataset in "${DATASETS[@]}"; do
    echo "########################################################"
    echo "STARTING DATASET: $dataset"
    echo "########################################################"

    # 第二层循环：遍历种子 2021 到 2025
    for seed in {2025..2025}; do
        echo "  =================================================="
        echo "  Dataset: $dataset | Current Seed: $seed"
        echo "  =================================================="

        # 第三层循环：遍历模型
        for model in "${MODELS[@]}"; do

            # --- 核心逻辑判断 ---
            if [ "$model" == "NDCC" ]; then
                # === NDCC 特殊处理 ===
                # NDCC 论文通常跑 Strategy 1 和 2
                # 如果你只需要跑其中一种，请注释掉不需要的那一行

                echo "    [Running] Model: NDCC (Strategy 1) ..."
                python TLN.py --model NDCC --dataset "$dataset" --seed "$seed" --strategy 1 --device "$DEVICE" --network "cnn"

#                echo "    [Running] Model: NDCC (Strategy 2) ..."
#                python TLN.py --model NDCC --dataset "$dataset" --seed "$seed" --strategy 2 --device "$DEVICE"

            else
                # === 其他模型 ===
                # 统一强制使用 Strategy 3
                echo "    [Running] Model: $model (Strategy 3) ..."
                python TLN.py --model "$model" --dataset "$dataset" --seed "$seed" --strategy 3 --device "$DEVICE" --network "cnn"
            fi

        done
        echo "" # 空行，为了日志美观
    done
done

echo "########################################################"
echo "ALL EXPERIMENTS FINISHED!"
echo "########################################################"