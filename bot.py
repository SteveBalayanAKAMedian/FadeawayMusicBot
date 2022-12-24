from lyricsgenius import Genius
import json
import codecs
import os
import re
from collections import Counter
import numpy as np
import json
import pickle
import logging
from aiogram import Bot, Dispatcher, executor, types


class NGramModel:
    def __init__(self, model_path, input_dir='default_stream_stdin', n=-1):
        self.model_path = model_path
        self.input_dir = input_dir
        self.frequency = dict()
        if n == -1:
            self.n = json.load(open('config.json'))['n']
        else:
            self.n = n

    @staticmethod
    def __make_text_nice(s):
        s = re.sub(r'\n', ' ', s)
        s = re.sub(r' +', ' ', s)
        s = re.sub(r'[^a-zA-Z0-9\' ]', '', s)
        s = s.lower()
        if s[0] == ' ':
            s = s[1:]
        return s

    def fit(self):
        uploaded_texts = []
        if self.input_dir == 'default_stream_stdin':
            s = input()
            uploaded_texts.append(NGramModel.__make_text_nice(s))
        else:
            try:
                files = os.listdir(self.input_dir)
            except Exception as e:
                print('cannot get training data, smth is amiss with the input_dir')
                print(str(e))
                print('fit terminated')
                return
            for file in files:
                if '.txt' in file:
                    try:
                        with open(self.input_dir + '\\' + file, encoding='utf-8') as f:
                            s = f.read()
                            s = NGramModel.__make_text_nice(s)
                            uploaded_texts.append(s.split())
                    except Exception as e:
                        print('something went wrong with the file named ', file)
                        print(str(e))
                        print('gonna skip it and check other files')

        n = len(uploaded_texts)
        print('uploaded {} text files'.format(n))
        for i in range(n):
            m = len(uploaded_texts[i])
            for j in range(m - self.n - 1):
                pref = [''] * self.n
                for k in range(j, j + self.n):
                    pref[k - j] = uploaded_texts[i][k]
                pref = tuple(pref)
                if self.frequency.get(pref) is None:
                    self.frequency[pref] = Counter()
                self.frequency[pref][uploaded_texts[i][j + self.n]] += 1

        cnt = len(self.frequency)
        tmp = 0
        for pref in self.frequency.keys():
            counter = self.frequency[pref].most_common()
            m = len(counter)
            res = [('', 0) for i in range(m)]
            cnt_suffix = 0
            for i in range(m):
                cnt_suffix += counter[i][1]
            for i in range(m):
                res[i] = (counter[i][0], counter[i][1] / cnt_suffix)
            self.frequency[pref] = res
            if len(self.frequency[pref]) > 1:
                tmp += 1
        print('in {cnt} prefixes found {tmp} prefixes following by multiple words'.format(cnt=cnt, tmp=tmp))
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self, f)
        except Exception as e:
            print('cannot save model')
            print(str(e))
            return
        print('Successfully generated and saved to {}'.format(self.model_path))

    def load(self):
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.frequency = data.frequency
                self.n = data.n
                self.model_path = data.model_path
                self.input_dir = data.input_dir
        except Exception as e:
            print('cannot load the model')
            print(str(e))

    def generate(self, length, prefix=''):
        np.random.seed = len(self.frequency) * len(self.model_path) * len(self.input_dir) * self.n

        if prefix == '':
            prefix = list(self.frequency.keys())[np.random.randint(len(self.frequency.keys()))]
        else:
            print(prefix)
            prefix = tuple(NGramModel.__make_text_nice(prefix).split())
            if prefix not in self.frequency.keys():
                print('given prefix not found in train data, unfortunately, will have to use a random prefix')
        res = list(prefix)
        while len(res) != length:
            if prefix not in self.frequency.keys():
                # если такой префикс не существует, рандомим
                # очень грубое решение, но результат получается смешной
                prefix = list(self.frequency.keys())[np.random.randint(len(self.frequency.keys()))]
            l = len(self.frequency[prefix])
            words = [self.frequency[prefix][i][0] for i in range(l)]
            probabilities = [self.frequency[prefix][i][1] for i in range(l)]
            next_word = np.random.choice(words, 1, p=probabilities)[0]
            res.append(next_word)
            prefix = prefix[1:] + tuple([next_word])
        ans = ''
        for i in res:
            ans += i + ' '
        ans = ans[:-1]
        return ans

CONFIG = json.load(open('config.json'))
API_TOKEN = CONFIG["tgbot_token"]

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)