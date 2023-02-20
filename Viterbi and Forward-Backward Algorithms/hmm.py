from typing import Pattern, Union, Tuple, List, Dict, Any

import numpy as np
import numpy.typing as npt


Numeric = Union[float, int, np.number, None]


"""
Global list of parts of speech
"""
POS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
       'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']
"""
Utility functions for reading files and sentences
"""
def read_sentence(f):
    sentence = []
    while True:
        line = f.readline()
        if not line or line == '\n':
            return sentence
        line = line.strip()
        word, tag = line.split("\t", 1)
        sentence.append((word, tag))

def read_corpus(file):
    f = open(file, 'r', encoding='utf-8')
    sentences = []
    while True:
        sentence = read_sentence(f)
        if sentence == []:
            return sentences
        sentences.append(sentence)


"""
Supervised learning
Param: data is a list of sentences, each of which is a list of (word, POS) tuples
Return: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities} 
"""
def learn_model(data:List[List[Tuple[str]]]
                ) -> Tuple[npt.NDArray, npt.NDArray, Dict[str,npt.NDArray]]:
    X0 = np.zeros(len(POS))
    Tprop = np.zeros((len(POS),len(POS)))#col to row
    numT = np.zeros(len(POS))
    prob0 = {}
    for sent in data:
        X0[POS.index(sent[0][1])]+=1
        for i in range(len(sent)-1):
            bPOS = POS.index(sent[i][1])
            numT[bPOS]+=1
            aPOS = POS.index(sent[i+1][1])
            Tprop[aPOS][bPOS] += 1
    for sent in data:
        for i in range(len(sent)):
            w = sent[i][0]
            p = POS.index(sent[i][1])
            if not w in prob0:
                prob0[w] = np.zeros(len(POS))
            prob0[w][p] = prob0[w][p]+1

    #Normalize
    X0 = X0/sum(X0)
    Tprop = Tprop/numT
    for i in prob0.keys():
        prob0[i] = prob0[i]/sum(prob0[i])
    return X0, Tprop, prob0


"""
Viterbi forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: m, 1D array; pointers, 2D array
"""
def viterbi_forward(X0:npt.NDArray,
                    Tprob:npt.NDArray,
                    Oprob:Dict[str,npt.NDArray],
                    obs:List[str]
                    ) -> Tuple[npt.NDArray, npt.NDArray]:
    m = X0
    pointers = []
    for k in range(len(obs)):
        m2 = []
        p2 = []
        for i in range(len(Tprob)):
            h_pos = 0
            for j in range(len(Tprob[0])):
                if Tprob[i][j]*m[j] > Tprob[i][h_pos]*m[h_pos]:
                    h_pos = j
            m2.append(Tprob[i][h_pos]*m[h_pos])
            p2.append(h_pos)
        pointers.append(p2)
        m2 = np.array(m2)
        #m2 is now m'
        O_t = np.zeros((len(POS),len(POS)))
        obs2 = np.ones(len(POS))
        if obs[k] in Oprob:
            obs2 = Oprob[obs[k]]
        for i in range(len(O_t)):
            O_t[i][i] = obs2[i]
        m = np.matmul(O_t,m2)

    return m, np.array(pointers)


"""
Viterbi backward algorithm
Param: m, 1D array; pointers, 2D array
Return: List of most likely POS (strings)
"""
def viterbi_backward(m:npt.NDArray,
                     pointers:npt.NDArray
                     ) -> List[str]:
    most_like = []
    current = np.where(np.amax(m)==m)[0][0]
    for i in range(len(pointers)-1,-1,-1):
        most_like.insert(0,POS[current])
        current = pointers[i][current]
    return most_like


"""
Evaluate Viterbi by predicting on data set and returning accuracy rate
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of (word,POS) pairs
Return: Prediction accuracy rate
"""
def evaluate_viterbi(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     data:List[List[Tuple[str]]]
                     ) -> float:
    succ = 0
    total = 0
    for i in range(len(data)):
        words = []
        pos = []
        for j in range(len(data[i])):
            words.append(data[i][j][0])
            pos.append(data[i][j][1])
        m, pointers = viterbi_forward(X0, Tprob, Oprob, words)
        predict_pos = viterbi_backward(m,pointers)
        for k in range(len(pos)):
            if pos[k] == predict_pos[k]:
                succ+=1
            total+=1

    return succ/total


"""
Forward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings)
Return: P(XT, e_1:T)
"""
def forward(X0:npt.NDArray,
            Tprob:npt.NDArray,
            Oprob:Dict[str,npt.NDArray],
            obs:List[str]
            ) -> npt.NDArray:
    a = X0
    for i in range(len(obs)):
        a2 = np.matmul(Tprob, a)#a'
        obs2 = np.ones(len(POS))
        if obs[i] in Oprob:
            obs2 = Oprob[obs[i]]
        O_t = np.zeros((len(POS),len(POS)))
        for k in range(len(O_t)):
            O_t[k][k] = obs2[k]
        a = np.matmul(O_t,a2)
    return a


"""
Backward algorithm
Param: Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(e_k+1:T | Xk)
"""
def backward(Tprob:npt.NDArray,
             Oprob:Dict[str,npt.NDArray],
             obs:List[str],
             k:int
             ) -> npt.NDArray:
    Tprob = np.transpose(Tprob)
    b = np.ones(len(POS))
    j = len(obs)-1
    while j >= k:
        obs2 = np.ones(len(POS))
        if obs[j] in Oprob:
            obs2 = Oprob[obs[j]]
        O_t = np.zeros((len(POS),len(POS)))
        for i in range(len(O_t)):
            O_t[i][i] = obs2[i]
        b2 = np.matmul(O_t, b)
        b = np.matmul(Tprob, b2)
        j-=1
    return b


"""
Forward-backward algorithm
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; obs, list of words (strings); k, timestep
Return: P(Xk | e_1:T)
"""
def forward_backward(X0:npt.NDArray,
                     Tprob:npt.NDArray,
                     Oprob:Dict[str,npt.NDArray],
                     obs:List[str],
                     k:int
                     ) -> npt.NDArray:
    
    a = forward(X0, Tprob,Oprob,obs[:k])
    a = a/sum(a)
    b = backward(Tprob, Oprob, obs, k)
    c = a*b
    c = c/sum(c)
    return c


"""
Expected observation probabilities given data sequence
Param: P(X0), 1D array; Tprob, 2D array; Oprob, dictionary {word:probabilities}; data, list of lists of words
Return: New Oprob, dictionary {word:probabilities}
"""
def expected_emissions(X0:npt.NDArray,
                       Tprob:npt.NDArray,
                       Oprob:Dict[str,npt.NDArray],
                       data:List[List[str]]
                       ) -> Dict[str,npt.NDArray]:
    hidden = {}
    g_total = np.zeros(len(POS))
    for i in range(len(data)):
        for k in range(len(data[i])):
            g = forward_backward(X0,Tprob,Oprob, data[i], k+1)
            g_total+=g
            if data[i][k] in hidden:
                hidden[data[i][k]]=hidden[data[i][k]]+g
            else:
                hidden[data[i][k]]=g
    for i in hidden.keys():
        hidden[i] = hidden[i]/g_total
        hidden[i] = hidden[i]/sum(hidden[i])
    return hidden


if __name__ == "__main__":
    train = read_corpus('train.upos.tsv')
    test = read_corpus('test.upos.tsv')
    X0, T, O = learn_model(train)
    print("Train accuracy:", evaluate_viterbi(X0, T, O, train))
    print("Test accuracy:", evaluate_viterbi(X0, T, O, test))

    obs = [[pair[0] for pair in sentence] for sentence in [test[0]]]
    Onew = expected_emissions(X0, T, O, obs)
    print(Onew)
