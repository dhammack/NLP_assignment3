#assignment 3 - Daniel Hammack
import string
import math
import time
from collections import defaultdict
from itertools import *
import cPickle as pic

t = defaultdict(float) #t(f|e), maps: (foreign_word,english_word) -> probability
trainset = list()
d_cache = {} #holds denominator for the delta function.
q = defaultdict(float) #q(j|i,l,m), maps: (j, i, en_len, es_len) -> prob
#bi_cts = defaultdict(float)
#uni_cts = defaultdict(float)

#simple bigram language model
def r(wi_prev, wi):
    uni = uni_cts[wi_prev]
    bi = bi_cts[(wi_prev,wi)]
    #if uni < 4:
    #    uni = uni_cts['RARE']
    if uni == 0.0:
        return 0
    return bi / uni

def estimate_params(en_file, es_file):
    global t, q
    print 'loading data...'
    #pickle_load('pickled_data.pickle')
    init_step(en_file, es_file)
    init_prob2()

    print 'maximizing t expectations...'
    EM_t_only(iterations = 5)
    
    print 'maximizing t&q expectations...'
    EM_both(iterations = 5)
    
    print 'saving...'
    pickle_save(location = 'pickled_data_4.pickle')
    
    print 'done!'


def init_prob2():
    global q
    #initialize q
    for (en_sent, es_sent) in trainset:
        L,m = len(en_sent), len(es_sent)
        p = 1.0/float(L+1.0)
        for i in xrange(m):
            for j in xrange(L+1):
                q[(j,i,L,m)] = p

#part 2
def EM_both(iterations):
    global t, q
    c = defaultdict(float)
    n = len(trainset)
    for s in xrange(0,iterations):
        print 'iteration:', s+1,'of',iterations
        c.clear()
        d_cache.clear()
        for (en_sent, es_sent) in trainset:
            L,m = len(en_sent), len(es_sent)
            for i,es_word in enumerate(es_sent):
                for j,en_word in enumerate(en_sent):

                    d = delta(en_sent, en_word, es_word, j, i, L, m)
                    c[(en_word, es_word)] += d
                    c[en_word] += d
                    c[(j,i,L,m)] += d
                    c[(i,L,m)] += d

        for (es_word,en_word) in t:
            t[(es_word,en_word)] = c[(en_word,es_word)] / c[en_word]

        for (j,i,L,m) in q:
            q[(j,i,L,m)] = c[(j,i,L,m)] / c[(i,L,m)]

#part 1
def EM_t_only(iterations):
    global t
    c = defaultdict(float)
    n = len(trainset)
    for s in xrange(0,iterations):
        print 'iteration:', s+1,'of',iterations
        c.clear()
        d_cache.clear()
        for (en_sent, es_sent) in trainset:
            L,m = len(en_sent), len(es_sent)
            for i,es_word in enumerate(es_sent):
                en_prev = '*'
                for j,en_word in enumerate(en_sent):

                    d = delta(en_sent, en_word, es_word, j, i, L, m)
                    c[(en_word, es_word)] += d
                    c[en_word] += d
                    en_prev = en_word
                    
        for (es_word,en_word) in t:
            t[(es_word,en_word)] = c[(en_word,es_word)] / c[en_word]

def delta(en_sent, en_word, es_word, j, i, L, m):
    global d_cache
    #h = hash_list(en_sent)
    if (en_sent, es_word) not in d_cache:

        d_cache[(en_sent, es_word)] = (sum(t[(es_word, e)] * q[(jj,i,L,m)]
                for jj,e in enumerate(en_sent))) + L
        
    return ((t[(es_word,en_word)]*q[(j,i,L,m)] + 1)
             / d_cache[(en_sent, es_word)])

   
#save the t params, en and es wordlists.
def pickle_save(location):
    with open(location, 'wb') as f:
        pic.dump(t, f, -1)
        pic.dump(q, f, -1)
        #pic.dump(bi_cts, f, -1)
        #pic.dump(uni_cts, f, -1)

        
#load the t params, en and es wordlists.
def pickle_load(location):
    global t, q, uni_cts, bi_cts 
    with open(location, 'rb') as f:
        t = pic.load(f)
        q = pic.load(f)
        #bi_cts = pic.load(f)
        #uni_cts = pic.load(f)
        
#initializes t(f|e) = 1/n(e) forall e, loads each wordlist
def init_step(en_file, es_file):
    global t, trainset, uni_cts, bi_cts
    #uni_cts['RARE'] = 0
    en_wordlist, es_wordlist = set(), set()
    nlist = defaultdict(set) #maps: english_word -> list of foreign words
    with open(en_file, 'r') as f_en:
        with open(es_file, 'r') as f_es:
            for (en_line,es_line) in izip(f_en,f_es):
                en_words = en_line.replace('\n','').split(' ')
                es_words = es_line.replace('\n','').split(' ')
                en_words.insert(0, '') #null word
                en_wordlist.update(en_words)
                es_wordlist.update(es_words)
                #trainset is a list of all tuples in the training data.
                trainset.append( (tuple(en_words),tuple(es_words)))
                for e in en_words:
                    nlist[e].update(es_words)
                #    uni_cts[e] += 1

                #NEW: language model parameters
                #if len(en_words) == 0: continue
                #uni_cts['*'] += 1
                #bi_cts[('*',en_words[0])] += 1
                #for i in xrange(0,len(en_words)-1):
                #    bi_cts[(en_words[i],en_words[i+1])] += 1
                
                
    for e in en_wordlist:
        es_word_count = float(len(nlist[e]))
        for f in nlist[e]:
            t[(f,e)] = 1.0 / es_word_count

    #set up rare count.
    #for word, ct in uni_cts.iteritems():
    #    if ct < 4:
    #        uni_cts['RARE'] += ct
    #        uni_cts[word] = 0

    
#def hash_list(L):
#    if len(L) > 0:
#        if len(L) > 1:
#            return hash(L[0]) + hash(L[1]) + len(L)
#        else:
#            return hash(L[0] + hash(tuple(L)))
#    else:
#        return hash(tuple(L))
    
def predict_aligns(en_test, es_test, outpath):
    
    #load the pickled data; the EM algorithm should be done.
    print 'loading pickled data...'
    pickle_load('pickled_data_4.pickle')

    print 'making predictions...'  
    #make predictions, using argmax a_i over j=0 to l of t(f_i|e_j)
    with open(outpath, 'w') as f_out:
        with open(en_test, 'r') as f_en, open(es_test, 'r') as f_es:
            for k,(en_line, es_line) in enumerate(izip(f_en, f_es)):
                en_sent = en_line.replace('\n','').split(' ')
                es_sent = es_line.replace('\n','').split(' ')
                en_sent.insert(0,'')
                L,m = len(en_sent), len(es_sent)

                en_prev = '*' #language model.

                #for each foreign word
                for i,fword in enumerate(es_sent):
                    #a_i = argmax over english indices of t(f_i|e_j)
                    a_i, best, current, best_word = 0, 0.0, 0.0, ''
                    for j,eword in enumerate(en_sent):
                        #if r(en_prev, eword) == 0.0: continue
                        current = t[(fword,eword)]*q[(j,i,L,m)]#*r(en_prev, eword)
                        #print 'r', (en_prev, eword), ':' ,r(en_prev, eword)
                        #raw_input()
                        if current >= best:
                            a_i = j
                            best = current
                            best_word = eword

                    #print 'using:',en_prev,best_word
                    #raw_input()
                    en_prev = best_word
                    
                    f_out.write(str(k+1) + ' ' +
                                str(a_i) + ' ' +
                                str(i+1) +'\n')

    print 'done!'

    
todos = """
* part 3: model each as the foriegn lang and take the union of the alignments
* part 3: bigram language model didn't help.
* part 3: remember we're only predicting alignments. Introduce some preference
for adjacent words, perhaps?
"""
print todos
#estimate_params(en_file='corpus.en', es_file='corpus.es')
predict_aligns(en_test='dev.en',es_test='dev.es',
               outpath='dev_pt3_out_6.txt')
