# -*- coding: utf-8 -*-
"""
Created on Thu May 03 17:48:14 2018

@author: sky
"""

import os
import jieba
import numpy as np
import pandas as pd
#读入停用词
stopwords = []
for line in open('chinese_stopword.txt','r',encoding='utf-8'):
    stopwords.append(line.rstrip()) #去掉尾部的空白字符
    
def genDoc(path, stopwords):
    '''
    生成文本集二重列表，每一个文件分词之后的词语作为一个列表
    所有文件生成的词汇表构成Doc列表
    '''       
    Doc = []
    word_list = set()
    title_list = list()
    for f in os.listdir(path):
        if not os.path.isdir(f): #判断是否是文件夹 不是文件夹才打开
            title_list.append(f)
            with open(path + '/' + f, 'r', encoding='utf-8') as t:
                #将空格、换行符去除
                content = t.read().strip().replace('\n','')\
                        .replace(' ','').replace('\r','').replace('\t','') 
                #使用基于深度学习的FoolNltk分词
                seg_content = list(jieba.cut(content))
                seg_content_dropstop = []
                for word in seg_content:
                    if word not in stopwords:
                        seg_content_dropstop.append(word)
                        word_list.add(word)
            Doc.append(seg_content_dropstop)
    return Doc,word_list,title_list
    

def tf(word, doc):
    return doc.count(word) / len(doc)
    
def idf(word, Doc):
    count = 0
    for each_doc in Doc:
        if word in each_doc:
            count += 1        
    return np.log(len(Doc)/(count+1))

def getTFIDF(Doc, word_list, title_list):
    '''
        获取一个行号为文件名，列号为单词的DataFrame数据结构
    '''
    tfidf_matrix = np.zeros((len(title_list), len(word_list)))
    df_tfidf = pd.DataFrame(tfidf_matrix,columns= word_list)
    df_tfidf.index = title_list
    for index,each_doc in enumerate(Doc):
        for each_word in each_doc:
            df_tfidf.loc[title_list[index],each_word] = tf(each_word,each_doc) * idf(each_word,Doc)
    return df_tfidf

def getTopk(df_tfidf, topk = 3):
    for doc_index, doc_words in df_tfidf.iterrows():
        print(doc_index)
        print(doc_words.sort_values(ascending=False)[0:topk])
        
if __name__ == "__main__":
    #测试
    Doc, word_list, title_list = genDoc(r'./data', stopwords)
    df_tfidf = getTFIDF(Doc, word_list, title_list)
    getTopk(df_tfidf, topk=5)

    