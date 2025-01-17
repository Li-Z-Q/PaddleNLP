# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export https_proxy=http://124.16.138.148:7890 http_proxy=http://124.16.138.148:7890 all_proxy=socks5://124.16.138.148:7890 no_proxy=localhost,127.0.0.0/8,*.local
curl https://ipinfo.io/
# CUDA_VISIBLE_DEVICES=0 nohup \
python3.10 -u /141nfs/lizhuoqun/PaddleNLP/slm/pipelines/examples/contrastive_training/evaluation/eval_mteb.py \
    --base_model_name_or_path /141nfs/lizhuoqun/hf_models/LLARA-passage-paddle \
    --output_folder /141nfs/lizhuoqun/PaddleNLP/slm/pipelines/examples/contrastive_training/evaluation/results/LLARA-passage-paddle \
    --task_name 'MSMARCOTITLE' \
    --eval_batch_size 8 \
    --max_seq_length 532 \
    --task_split test \
    --pooling_method last_8 \
    --add_bos_token 1 \
    --add_eos_token 0
