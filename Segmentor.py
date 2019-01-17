#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import log
from collections import defaultdict
import re
import pickle
import sys


class Segmentor():
    """中文分词工具 by thinkwee，基于最大概率切分、最长前向匹配和最长后向匹配三种算法

    Parameters
    ----------
    more : int, optional(default=1)
        是否使用额外的语料库训练，可以在程序所在目录下新建training_more.utf8并写入自定义的分词训练集来训练,
        否则使用自带的msr_training.utf8作为训练集

    Attributes
    ----------
    self.more : bool
        同 more

    self.dic_uni_ : defaultdict(int)
        一元语法词频词典

    self.dic_bi_ : defaultdict(int)
        二元语法词频词典

    """

    def __init__(self, more):
        self.more_ = more
        self.dic_uni_ = None
        self.dic_bi_ = None

    def build_dic(self, save):
        """得到文档中N元语法的词频词典并保存

        统计训练语料中一元语法和二元语法的词频
        训练语料用两个空格作为分词标记，未去除标点符号
        每一句话之前添加'$'作为开始符号

        :param save: int, optional(default=1)
            是否保存词频词典
        :return None:
        """

        dic_unigram = defaultdict(int)
        dic_bigram = defaultdict(int)

        if self.more_:
            file_training = open('./training_more.utf8', 'r', encoding='utf8')
        else:
            file_training = open('./msr_training.utf8', 'r', encoding='utf8')

        for line in file_training:
            word_list = re.split(r"  ", line)
            while '' in word_list:
                word_list.remove('')
            word_list.insert(0, '$')
            for idx, word in enumerate(word_list):
                # 处理边界情况
                if idx >= 1:
                    dic_bigram[word_list[idx - 1] + '|' + word] += 1

                dic_unigram[word] += 1

        self.dic_uni_ = dic_unigram
        self.dic_bi_ = dic_bigram

        print('Ngram dict built')

        if save:
            if 'pickle' in sys.modules:
                with open("./dic_uni.pkl", "wb") as f:
                    pickle.dump(dic_unigram, f)
                with open("./dic_bi.pkl", "wb") as f:
                    pickle.dump(dic_bigram, f)
                print("Ngram dict saved")
            else:
                print("can not save the dict, try 'pip install pickle'")

        return None

    def load_dict(self):
        """读取词频词典

        读取一元和二元语法词频词典

        :return None:
        """
        if 'pickle' in sys.modules:
            try:
                self.dic_uni_ = pickle.load(open('./dic_uni.pkl', 'rb'))
                self.dic_bi_ = pickle.load(open('./dic_bi.pkl', 'rb'))
                print("dict load")
            except FileNotFoundError:
                print("file not found,run build_dict() first")
        else:
            print("can not load the dict, try 'pip install pickle'")

        return None

    def get_dict(self, n):
        """获取词典

        返回词频词典

        :param n: int
            指定返回n元语法词频词典
        :return dict: defaultdict(int)
            根据n返回n元语法词频词典
        """
        if n == 1:
            return self.dic_uni_
        elif n == 2:
            return self.dic_bi_
        else:
            print("n must be 1 or 2")
            return None

    def _run_max_prob(self, sentence):
        """分句进行最大概率切分

        将每个输入按句拆分，对每一句进行最大概率切分，再拼起来返回

        :param sentence: str
            待切分文段
        :return sentence_seg: list
            切分之后的文段
        """
        sentence_seg = []
        punc_list = []
        len_sentence = len(sentence)

        # 获得所有分割句子的标点符号
        idx = 0
        while idx < len_sentence:
            if sentence[idx] in ['。', '？', '！']:
                punc_list.append(sentence[idx])
            idx += 1

        # 按标点符号分割句子
        sentence_list = re.split("。|？|！", sentence)
        if sentence_list[-1] == "":
            sentence_list = sentence_list[:-1]

        # 逐句进行最大概率切分，每一句包括跟在后面的标点符号
        for idx, s in enumerate(sentence_list):
            if idx < len(punc_list):
                s += punc_list[idx]
            line_seg = self._max_prob(s)
            sentence_seg += line_seg

        return sentence_seg

    def _max_prob(self, sentence):
        """对单句最大概率切分

        构造所有可能切分的有向无环图，使用动态规划，算出概率最大的切分

        :param sentence: str
            待切分句子
        :return sentence_seg: list
            切分好的句子
        """

        # 统计词典大小，用于平滑
        vocab_size = len(self.dic_uni_)

        # 构建前缀词典
        sentence_seg = []
        dict_prefix = defaultdict(int)
        sentence = '$' + sentence
        len_sentence = len(sentence)
        for i in range(len_sentence):
            for j in range(max(0, i - 10), i):
                if sentence[j:i + 1] in self.dic_uni_:
                    dict_prefix[sentence[j:i + 1]] += self.dic_uni_[
                        sentence[j:i + 1]]

        # 构建有向无环图
        DAG = []
        for i in range(len_sentence):
            list_i = []
            for item in dict_prefix:
                if item[0] == sentence[
                        i] and i + len(item) - 1 < len_sentence and sentence[
                            i:i + len(item)] in dict_prefix:
                    if dict_prefix[sentence[i:i + len(item)]] > 0:
                        list_i.append(i + len(item) - 1)
            if i not in list_i:
                list_i.insert(0, i)
            DAG.append(list_i)

        # 遍历有向无环图，转换成以句中位置为下表的转移矩阵
        transform = [[-10e6 for _ in range(len_sentence)]
                     for _ in range(len_sentence)]
        for start_pos, item in enumerate(DAG):
            for end_pos in item:
                start_word = sentence[start_pos:end_pos + 1]
                if end_pos >= len_sentence - 1:
                    continue
                for end_span in DAG[end_pos + 1]:
                    end_word = sentence[end_pos + 1:end_span + 1]

                    # 计算bigram概率，加1平滑
                    if self.dic_uni_[start_word] == 0:
                        prob = log(1 / (2 * vocab_size))

                    else:
                        prob = log(
                            (self.dic_bi_[start_word + '|' + end_word] + 1) /
                            (self.dic_uni_[start_word] + vocab_size))

                    if prob > transform[end_pos][end_span]:
                        transform[end_pos][end_span] = prob

                        # 打印测试
                        # print("%s %f %d" % (start_word + ' ' + end_word, prob, dic_bigram[start_word + '|' + end_word]))
                        # print(transform[end_pos][end_span], end_pos, end_span)

        # 根据转移矩阵进行动态规划
        dp_transform = [-10e6 for _ in range(len_sentence)]
        dp_transform[0] = 0
        back = [-1 for _ in range(len_sentence)]

        for i in range(1, len_sentence):
            for j in range(i):
                if dp_transform[j] + transform[j][i] > dp_transform[i]:
                    dp_transform[i] = dp_transform[j] + transform[j][i]
                    back[i] = j

        # 根据回溯数组找到切分位置，切分单词
        idx = len_sentence - 1
        while idx > 0:
            tmp = idx
            idx = back[idx]
            sentence_seg.append(sentence[idx + 1:tmp + 1])

        # 倒序添加切分词，因此需要对列表进行一次反转
        sentence_seg.reverse()

        # 删除开始符号
        if '$' in sentence_seg:
            sentence_seg.remove('$')

        return sentence_seg

    def _forward_max_match(self, sentence):
        """前向最大匹配

        从左往右分割，尽可能分割长的语段作为词

        :param sentence: str
            待切分句子
        :return sentence_seg: list
            分词后的句子
        """
        start = 0
        len_sentence = len(sentence) + 1
        sentence_seg = []
        oov = ""  # 暂存未登录词

        # 循环每一个位置
        while start < len_sentence:
            has_seg = False
            span = 1

            # 前向最大匹配
            while sentence[
                    start:start +
                    span] in self.dic_uni_ and start + span < len_sentence:
                span += 1
                has_seg = True

            # 如果该位置往后做了切分
            if has_seg:
                # 如果之前跳过了一些位置，存入了未登录词，则将未登录词加入词典，并清空未登录词缓存
                if oov != "":
                    sentence_seg.append(oov)
                oov = ""

                span -= 1  # 由于while循环，分词长度多加了1，需减去
                sentence_seg.append(sentence[start:start + span])  # 加入分词
                start += span  # 跳到下一位置

            # 如果从该位置开始往后无法再分词
            else:
                oov = oov + sentence[start:start + 1]  # 将该位置单字加入未登录词缓存
                start += 1  # 跳过该位置
        if oov != "":
            sentence_seg.append(oov)
        return sentence_seg

    def _backward_max_match(self, sentence):
        """后向最大匹配

        从右往左分割，尽可能分割长的语段作为词

        :param sentence: str
            待切分句子
        :return sentence_seg: list
            切分好的句子
        """

        len_sentence = len(sentence)
        start = len_sentence - 1
        sentence_seg = []
        oov = ""  # 暂存未登录词

        # 循环每一个位置
        while start >= 0:
            has_seg = False
            span = 1

            # 后向最大匹配
            while sentence[start -
                           span:start] in self.dic_uni_ and start - span >= 0:
                span += 1
                has_seg = True

            # 如果该位置往前做了切分
            if has_seg:
                # 如果之后跳过了一些位置，存入了未登录词，则将未登录词加入词典，并清空未登录词缓存
                if oov != "":
                    sentence_seg.append(oov[::-1])
                oov = ""

                span -= 1  # 由于while循环，分词长度多加了1，需减去
                sentence_seg.append(sentence[start - span:start])  # 加入分词
                start -= span  # 跳到下一位置

            # 如果从该位置开始往后无法再分词
            else:
                oov = oov + sentence[start - 1:start]  # 将该位置单字加入未登录词缓存
                start -= 1  # 跳过该位置
        sentence_seg.reverse()
        return sentence_seg

    def seg(self, method, file_path):
        """

        :param method: str, 'FMM' | 'BMM' | 'MP' ,default='MP'
            选择切分方法
        :param file_path: str, default = "./test.txt"
            指定待切分文档的路径
        :return None:
        """
        answers = []

        # 根据method选择方法进行分词
        with open(file_path, 'r', encoding='GBK') as f:
            if method == 'FMM':
                for line in f:
                    line_seg = ' '.join(self._forward_max_match(line))
                    answers.append(line_seg)
            elif method == 'BMM':
                for line in f:
                    line_seg = ' '.join(self._backward_max_match(line))
                    answers.append(line_seg + '\n')
            elif method == 'MP':
                for line in f:
                    line_seg = ' '.join(self._run_max_prob(line))
                    answers.append(line_seg)

        # 分词结果写入文件
        with open('./answers.txt', 'w', encoding='GBK') as f:
            for answer in answers:
                f.write(answer)

        print("file seg complete, saved in ./answers.txt")
        return
