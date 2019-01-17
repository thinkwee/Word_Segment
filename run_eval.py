#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

# 运行脚本，测试结果写入./score.ut8
status = os.system("perl ./icwb2-data/scripts/score \
           ./icwb2-data/gold/msr_training_words.utf8 \
           ./icwb2-data/gold/msr_test_gold.utf8 \
           ./icwb2-data/answers.utf8 > score.ut8")

# 打印测试结果
if status == 0:
    print("test complete")
    result = os.system("tail -14 score.ut8")
    print(result)
else:
    print("test wrong")
