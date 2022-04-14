import pandas as pd
from functools import reduce
from itertools import product

def nucleotide_type(k):
    z = []
    for i in product('ACGU', repeat = k):
        z.append(''.join(i))
    return z

def char_count(sequence,num,x):
    n = 0

    char = reduce(lambda x, y: [i + j for i in x for j in y], [['A', 'C', 'G', 'U']] * x)
    for i in range(len(sequence)):
        if sequence[i:i+x] == char[num]:
            n += 1
    return n/(len(sequence)-x+1)
def feature_336d(seq,k):
    list = []
    for i in range(4**k):
        list.append(char_count(seq,i,k))
    return (list)

def Sequence_replacement(sequ,k):
    sequen = [None]*len(sequ)
    for i in range(len(sequ)):
        s = sequ[i]
        sequen[i] = feature_336d(s,k)
    return sequen

if __name__ == '__main__':

    dataset = 'ncDR'
    k = 3
    miRNA = pd.read_csv(r'../Data/'+dataset+' miRNA sequrence.csv')
    seq = miRNA['sequence']
    name = miRNA['miRNA']
    feature_knf = Sequence_replacement(seq, k)

    kmer = []
    for i in range(len(name)):
        item = []
        item.append(name[i])
        for n in range(len(feature_knf[i])):
            item.append(str(feature_knf[i][n]))
        kmer.append(item)
    print(kmer[0])
    pd.DataFrame(kmer).to_csv(dataset+' miRNA kmer.csv', index=None, header=None)

    #pd.DataFrame({'miRNA':name,'kmer':feature_knf[:][:]}).to_csv(r'ncDR miRNA kmer1.csv', index=None, header=None)
