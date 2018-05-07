# -*- coding: utf-8 -*-
"""
Created on Fri May 04 16:58:15 2018

@author: sky
"""

'''
    代码部分参考snownlp的textrank实现
'''
import re
import jieba

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences

    

    
class KeywordTextRank(object):

    def __init__(self, docs):
        self.docs = docs
        self.words = {} #存储每个词以及与它相连的词
        self.vertex = {} #存储每个顶点以及textrank值
        self.d = 0.85  #浮点系数
        self.max_iter = 200 #最大迭代次数
        self.min_diff = 0.001 ##当变化最大的一次，仍然小于某个阈值时认为可以满足跳出条件，不用再循环指定的次数
        self.top = [] #排序之后的word列表

    def solve(self):
        #构建用词典存的图 每个词与他所相连的词构成一个集合
        for doc in self.docs: #遍历文档中的每一个句子
            que = [] #队列
            for word in doc: #遍历一个句子中的每一个词
                if word not in self.words:
                    self.words[word] = set()
                    self.vertex[word] = 1.0 #初始时候的textrank值
                que.append(word)
                if len(que) > 5: #以5为窗口进行划窗选词
                    que.pop(0)
                for w1 in que:
                    for w2 in que:
                        if w1 == w2:
                            continue
                        #在一个窗口中的任两个单词对应的节点之间存在一个无向边
                        self.words[w1].add(w2) 
                        self.words[w2].add(w1) 
        
        #迭代textrank值，上面初始化为1
        for _ in range(self.max_iter):
            m = {} #存储训练过程中每个词的TR值
            max_diff = 0 #最大差距
            tmp = filter(lambda x: len(self.words[x[0]]) > 0,
                         self.vertex.items()) #过滤掉没有一个连接词语的词 filter第一个参数为判断条件函数 第二个参数为迭代器
            #得到的tmp是一个遍历的(单词, TR值) 元组数组
            #按从小到大排序 可以加快收敛速度？
            tmp = sorted(tmp, key=lambda x: x[1] / len(self.words[x[0]]))  #x[1]表示这个词的TR值 len(words[x[0]]表示该词的出度) 
            for k, v in tmp:  #每一轮遍历所有节点
                for j in self.words[k]: #遍历单词k所连接的词
                    if k == j:
                        continue
                    if j not in m: #第一次出现的词
                        m[j] = 1 - self.d  
                    m[j] += (self.d / len(self.words[k]) * self.vertex[k]) #S(V^i) = (1-d) + d* sum(1/out(V^j) * S(V^j))
            for k in self.vertex: #训练过程中的TR值与最终存在vertex中的TR值做对比
                if k in m and k in self.vertex:
                    if abs(m[k] - self.vertex[k]) > max_diff:
                        max_diff = abs(m[k] - self.vertex[k])
            self.vertex = m #更新TR值
            if max_diff <= self.min_diff: #误差超过阈值 提前停止迭代
                break
        self.top = list(self.vertex.items())
        self.top = sorted(self.top, key=lambda x: x[1], reverse=True) #按从大到小排序

    def top_index(self, limit):
        return list(map(lambda x: x[0], self.top))[:limit]

    def top(self, limit):
        return list(map(lambda x: self.docs[x[0]], self.top))

text = '''
如果你是一位教师，那么不管你的工作单位是高中、大学还是职业培训等教育机构，你都能在MOOC上找到对学生有用的内容。近期许多MOOC实验项目的目标都是建设一个课堂教学支持系统。我们根据研究结果、采访和我们参与的课程整理出以下建议，希望能够指导教师将MOOC的经验和资源运用到传统课堂教学中。
1900年挪威特隆赫姆的科学课，图片来自WikiMedia。
1．善用电子设备
学生们都具备参与在线课堂的工具，而教师也可以利用自己的设备参与线上社交。如果你是一名学生，就把这篇文章转给你的老师吧。让老师看看怎样用智能手机和笔记本电脑记录你的活动，并迅速回复评论；看看你用哪些手机应用查看地图、搜索信息和购物；展示你的微博、人人和网盘，还有其他人留下的评论。最后，告诉老师，如果他们要开在线课程，并且需要学生助手，别忘了叫上你。
2．让课堂更广阔
MOOC的成功证明，相比课堂教学，许多学生更喜欢在线学习。随着智能手机、平板电脑和笔记本电脑在学生族中的流行，学生们有了与朋友、家人和老师交流的新方式。显然，在头脑清醒、精神集中的时候，看课程视频、做练习和参与课堂讨论的效率都更高。在MOOC，这意味着一个印度学生和一个新汉普郡的学生能够达到同一水平。因此在传统课堂中，我们也要让那些在足球训练场上、在滑雪度假村和在家养病的学生能够和坐在课时第一排的学生学到同样多的东西。在线工具能够让课堂不再受限于校园，学生们也能够获得更加美好而难忘的学习体验。
3．参与是成功之母
MOOC研究显示，成功的学生都会充分利用所有的在线资源，例如视频、测试、课堂讨论和由其他学生编写的课程维基百科。对于课堂教学，教师可以建立在线平台，让学生在课堂之外也能参与到学校中。这种交流可以只是发一条微博或者公告，邀请学生对备课提出建议。邀请学生把能够帮助他们复习所学知识的相关文章或图片发布到网上。只要利用这些小诀窍，大家就都能看到谁能跟上课程进度，进而鼓励他们或者提出建议。在课堂上可以表扬在网络上活跃的学生，或者解答学生在网上提出的问题。
4．建立网络社区
为了活跃课堂，你需要在网络上建一个“据点”，也就是说，需要建立一个网站，为学生提供课件，提醒任务完成时间，并且让学生之间建立联系。假设你是一位老师，并且已经有了自己的注册、评分和考勤系统，那就把这些运用到网络上吧。我们的目标是为学生建立一个平台，让他们能在现有的网络社交活动、与班上同学的交流和课堂作业之间自由转换。
W·伊恩-欧伯恩(W. Ian O’Byrne)是纽黑文大学的教育技术助教，他建议利用谷歌协作平台实现这个目的。他指出，学生们都有自己的移动设备，并且希望有一个“够酷而且适合聚会”的地方学习。只把课程都放到网上是不够的，需要精心设计漂亮的页面，并且这个网页要方便学生进行互动。
他说：“用好谷歌协作平台的关键仅仅是迈出第一步。把它当作一项工程，认真编辑，打造你自己的在线学习空间，就像每年组织一批学生一样。你可以在每个新学年调整课程内容，不断建设升级，把你的数字学习社区建设得越来越好。”
MOOC是学校的一种新形式，欧伯恩建议在起步的时候，先为每门课程的课件加上指针，再利用软件工具，就可以轻松根据学生的学习进度添加课程。他希望，在学生使用在线社区的同时，教师也能发现参与在线社区的方式。
5．从线上到线下
MOOC的一个缺陷就是无法组建高效的学习小组，而教师在这方面可以大有作为。当学生们看到其他同学更新了课程内容，他们就知道谁掌握了所学的知识，从而邀请这些同学合作完成任务，或向他们请教。我经常向教师们介绍这个例子：我在Google+圈子里发了一条信息，例如“明天我们会讨论矛盾冲突在吸引读者注意力方面的作用。今晚，在你回家的路上，拍一张照片或一段录像。用文字介绍你的见闻，以证明这个观点，并邀请其他同学参与讨论。”我收到的作业包括交通堵塞，猫狗对峙，被泡在水里的花园以及足球训练中的射门。第二天，学生们就可以归纳整理前一天晚上在网络上收集到的评论了。
6．用好你的相机
随着智能手机和平板电脑的普及，相机也变得无处不在，而且分享照片也越来越简单。MOOC的明星教授说，把45分钟的讲座变成10分钟一段的视频让他们被迫“升级课程”。不是每个老师都能通过这种方式吸引一批学生，但是他们可以参考这个经验，为课堂制作自己的视频，例如实地考察录像。让整个班都出去跑一趟可能不可行，但利用视频和照片，可以把考察点“带”到课室中来。利用智能手机耳机上配备的话筒，还可以为视频配上讲解，从而高效地用多个视频介绍完一个知识点。
将MOOC应用到传统课堂教学
随着大规模网络公开课的发展，教师可以考虑把在线教育的方法应用到自己的课堂教学中。MOOC的课程制作涉及比较复杂的技术，但使用这些课程几乎不费吹灰之力，而且成本也远远不及课程制作。没有加入edX或Coursera的大部分学校可以进行更多自创内容的尝试，就像自出版一样，这也是许多cMOOC的尝试。教师也可以向自己的目标努力。通过打开课堂，建立网络社区和制作教学视频，可以让更多的教师和学生享受到MOOC的投入带来的收益。
'''

sents = get_sentences(text)
doc = []
for sent in sents:
    #words = seg.seg(sent)
    words = list(jieba.cut(sent))
    #words = normal.filter_stop(words)
    doc.append(words)
keyword_rank = KeywordTextRank(doc)
keyword_rank
keyword_rank.solve()
for w in keyword_rank.top_index(5):
    print(w)