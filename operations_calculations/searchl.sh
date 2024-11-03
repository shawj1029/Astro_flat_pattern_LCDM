#!/bin/bash
#SBATCH --job-name=local            # 名称 local
#SBATCH --output=%x-%j.out          # 标准输出文件名，%x是作业名，%j是作业ID
#SBATCH --error=%x-%j.err           # 标准错误输出文件名
#SBATCH --nodes=1                   # 使用一个节点
#SBATCH --nodelist=comput1          # 使用node1节点
#SBATCH --ntasks=1                  # 在节点上启动一个任务
#SBATCH --gres=gpu:4                # 请求4个GPU
#SBATCH --partition=gpu             # 提交到GPU分区
#SBATCH --time=1024:00:00           # 运行时间为1024小时
#SBATCH --mem=70G                   # 请求70GB内存
#SBATCH --cpus-per-task=2           # 每个任务4个CPU核心

# 加载CUDA和Python环境
module load cuda/11.8.0-oneapi-2021.3.0-gjuht32

# 在skysearch下运行Python脚本
python /home/jiangxiaoqi/New/stage_3_parallel_program_5deg/parallel_program_5deg_l.py