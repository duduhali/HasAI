import re
from functools import reduce


'''返回切分后的词
>>>tokenize('Bob dropped the apple. Where is the apple?')
['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
'''
def tokenize(sent):
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

'''Parse stories provided in the bAbi tasks format
If only_supporting is true, only the sentences that support the answer are kept.
'''
def parse_stories(lines, only_supporting=False):
    data = []
    story = []
    for line in lines:
        # print(line) # b'1 John went to the bedroom.\n'
        # line = line.decode('utf-8').strip()
        # print(line) # 1 John moved to the bathroom.
        nid, line = line.split(' ', 1) #切割，只切第一个空格
        nid = int(nid) #序号
        if nid == 1:
            story = []
        if '\t' in line: #判断是否有制表符，Tab键
            q, a, supporting = line.split('\t')
            # print(q, a, supporting) #Where is the milk?  hallway 1 4
            q = tokenize(q)
            substory = None
            if only_supporting:
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

file = ['1 Mary got the milk there.','2 John moved to the bedroom.','3 Sandra went back to the kitchen.',
        '4 Mary travelled to the hallway.','5 Where is the milk? 	hallway	1 4',
        '6 John got the football there.','7 John went to the hallway.','8 Where is the football? 	hallway	6 7']
only_supporting=False
max_length=None
data = parse_stories(file, only_supporting=only_supporting)
print(data)
flatten = lambda data: reduce(lambda x, y: x + y, data) #定义一个函数，把字符串列表拼接在一起成功长字符串
data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
print(data)

for story, q, answer in data:
    print(story, q, answer)
    print(flatten(story))
