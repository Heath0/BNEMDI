import pandas as pd

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample))):
        label.append(1)
    return label

if __name__ == '__main__':
    data = pd.read_csv('PositiveSample_Train.csv',header=None)
    pdrug=data[1].values.tolist()
    pmiRNA=data[0].values.tolist()
    drugName = pd.read_csv(r'ALLDrug.csv').values.tolist()
    miRNAName = pd.read_csv(r'ALLMi.csv').values.tolist()

    milist = []
    druglist = []

    miss=[]

    for a in range(len(data)):
        for b in range(len(drugName)):
            if pdrug[a] == drugName[b][0]:
                t=0
                druglist.append(drugName[b][1])
                print(a, pdrug[a])
    pd.DataFrame(miss).to_csv(r'miss.csv', index=None, header=None)
    pd.DataFrame(druglist).to_csv(r'druglist.csv',index=None,header=None)
    print(len(druglist))

    for a in range(len(data)):
        for b in range(len(miRNAName)):
            if pmiRNA[a] == miRNAName[b][0]:
                milist.append(miRNAName[b][1])
                print(a, pmiRNA[a])
    print(len(milist))

    label = MyLabel(data)
    datadat = pd.DataFrame({'miRNA':milist,'drug':druglist,'label':label}).to_csv(r'datadat.csv',index=None,header=None)