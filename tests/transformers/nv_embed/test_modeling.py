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

import gc
import unittest

import paddle

from paddlenlp.transformers import NVEncodeModel


class NVEncodeModelIntegrationTest(unittest.TestCase):
    def test_model_tiny_logits(self):
        input_texts = [
            "This is a test",
            "This is another test",
        ]

        model_name_or_path = "NV-Embed-v1-paddle"
        model = NVEncodeModel.from_pretrained(
            model_name_or_path,
            tokenizer_path=model_name_or_path,
            dtype="float16",
            query_instruction="",
            document_instruction="",
        )
        with paddle.no_grad():
            out = model.encode_sentences(input_texts, instruction_len=0)

        print(out)
        """
        [[-0.00473404  0.00711441  0.01237488 ... -0.00228691 -0.01416779 -0.00429535]
         [-0.00343323  0.00911713  0.00894928 ... -0.00637054 -0.0165863 -0.00852966]]
        """

        del model
        paddle.device.cuda.empty_cache()
        gc.collect()
