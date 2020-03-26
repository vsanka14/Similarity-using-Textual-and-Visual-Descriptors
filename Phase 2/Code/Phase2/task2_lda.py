from parsers import parseUsersFile
from parsers import parseImageFile
from parsers import parseLocationFile
# from parsers import processMapToDataFrame
from scipy.spatial import distance

import gensim
from gensim import corpora
def task(val1 , val2 , val4) :

#val1 = int(input("Vector: "))
#val2 = int(input("k: "))
#val3 = int(input("top cont: "))
#val4 = input("ID: ")
# val1=1
# val2=4
# val3=4

# {k: v for v, k in enumerate(lst)}
    
    if(val1==1):
    	a = parseUsersFile()
    	# print(a.keys())
    	docs = list(a.keys())
    	mainlist=[]
    	for i in list(a.keys()):
    		sublist=[]
    		for k in list(a[i].keys()):
    			# print(a[i][k][0])
    			# sublist.extend([k for i in range(int(a[i][k][0]))])
    			sublist.extend([k] * int(a[i][k][0]))
    		mainlist.append(sublist)
    	# print(mainlist)
    	# break	
    	dictionary = corpora.Dictionary(mainlist)
    
    elif(val1==2):
    	a = parseImageFile()
    	docs = list(a.keys())
    	# print(a.keys())
    	mainlist=[]
    	for i in list(a.keys()):
    		sublist=[]
    		for k in list(a[i].keys()):
    			# print(a[i][k][0])
    			# sublist.extend([k for i in range(int(a[i][k][0]))])
    			sublist.extend([k] * int(a[i][k][0]))
    		mainlist.append(sublist)
    	dictionary = corpora.Dictionary(mainlist)
    
    elif(val1==3):
    	a = parseLocationFile()
    	docs = list(a.keys())
    
    	# print(a.keys())
    	mainlist=[]
    	for i in list(a.keys()):
    		sublist=[]
    		for k in list(a[i].keys()):
    			# print(a[i][k][0])
    			# sublist.extend([k for i in range(int(a[i][k][0]))])
    			sublist.extend([k] * int(a[i][k][0]))
    		mainlist.append(sublist)
    	dictionary = corpora.Dictionary(mainlist)
    
    # print(mainlist[0])
    # print(dictionary.doc2bow(mainlist[0]))
    # docs=list(set(docs))
    word2num = {k: v for v, k in enumerate(docs)}
    num2word = {v: k for v, k in enumerate(docs)}
    # print(Id_Mapping)
    
    # print(word2num)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in mainlist]
    
    #print(doc_term_matrix)
    # print(dictionary)
    Lda = gensim.models.ldamodel.LdaModel
    
    ldamodel = Lda(doc_term_matrix, num_topics=val2, id2word = dictionary, passes=1)
    
    print(str(ldamodel.print_topics(num_topics=val2)))
    
    def get_doc_topic(corpus, model): 
            doc_topic = list() 
            for doc in corpus: 
                doc_topic.append(model.__getitem__(doc, eps=0)) 
            return doc_topic 
    
    k = get_doc_topic(doc_term_matrix, ldamodel)
    
    dist_list = []
    
    for i in k:
    	temp = []
    	for j in i:
    		temp.append(j[1])
    	dist_list.append(temp)	
    
    req_id = word2num[val4]
    req_list = dist_list[req_id]
    all_dist_for_each = []
    
    for l, i in enumerate(dist_list):
    	dst = distance.euclidean(req_list, i)
    	subl = [num2word[l], dst]
    	all_dist_for_each.append(subl)
    
    # all_dist_for_each.sort(key=operator.itemgetter(1))
    final = sorted(all_dist_for_each, key=lambda x: x[1])
    print(final[:5])