import os

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path, encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):        # relaions.dict和entities.dict中的id都是按顺序排列的
            rel, id = line.strip().split("\t")
            index[rel] = id
            rev_index[id] = rel
    return index, rev_index


entity2id, id2entity = load_index(os.path.join('entity2id.txt'))
relation2id, id2relation = load_index(os.path.join('relation2id.txt'))

count = 0
count1 = 0
word_list = set() # 集合非重复
for entity_str in entity2id.keys(): # 对于每一个实体的real world description
    if "(" in entity_str and ")" in entity_str: # 如果real world description中包含括号
        count += 1 # 包含括号的计数
        begin = entity_str.find('(') # 左括号index
        end = entity_str.find(')') # 右括号index
        w1 = entity_str[:begin].strip() # 括号前string
        w2 = entity_str[begin+1: end] # 括号内string
        if w2 not in entity2id.keys(): # 括号内string不在实体集中
            print(w2)
            count1 += 1 # 括号内string不在实体集中计数
        word_list.add(w1) # 把包含括号的real world description的括号前string和括号内string都加入word_list
        word_list.add(w2)
    else:
        word_list.add(entity_str) # 如果real world description不包含括号直接加入word_list

num_word = len(word_list)

word2id = {word: id for id, word in enumerate(word_list)}
id2word = {id: word for id, word in enumerate(word_list)}

print("words num: {}, enity_num: {}".format(num_word, len(entity2id.keys())))
print(float(count)/len(entity2id.keys()))
print(float(count1)/float(count))

with open("word2id.txt", "w", encoding='utf-8') as f:
    for word in word2id.keys():
        f.write(word + "\t" + str(word2id[word])+'\n')

eid2wid = []
for id in range(len(id2entity.keys())): # 对于每一个原始实体集中的实体编号
    entity_str = id2entity[str(id)] # 获得原始实体string
    if "(" in entity_str and ")" in entity_str: # 如果包含括号
        count += 1
        begin = entity_str.find('(')
        end = entity_str.find(')')
        w1 = entity_str[:begin].strip() # 括号前string
        w2 = entity_str[begin+1: end] # 括号内string
        eid2wid.append([str(entity2id[entity_str]), "0", str(word2id[w1])])   # isA关系
        eid2wid.append([str(entity2id[entity_str]), "1", str(word2id[w2])])     # 隶属关系
    else:
        eid2wid.append([str(entity2id[entity_str]), "2", str(word2id[entity_str])])

with open("e-w-graph.txt", "w") as f:
    for line in eid2wid:
        f.write("\t".join(line)+'\n')




