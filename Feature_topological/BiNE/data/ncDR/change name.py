import pandas as pd

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample))):
        label.append(1)
    return label

if __name__ == '__main__':
    data = pd.read_csv('PositiveSample_Train.csv').values.tolist()
    drugName = pd.read_csv(r'ALLDrug.csv').values.tolist()
    miRNAName = pd.read_csv(r'ALLMi.csv').values.tolist()

    milist = []
    druglist = []

    for a in range(len(data)):
        print(a,data[a][1])
        for b in range(len(drugName)):
            if data[a][1] == drugName[b][0]:
                druglist.append(drugName[b][1])
    pd.DataFrame(druglist).to_csv(r'druglist.csv',index=None,header=None)
    print(len(druglist))

    for a in range(len(data)):
        print(a,data[a][0])
        for b in range(len(miRNAName)):
            if data[a][0] == miRNAName[b][0]:
                milist.append(miRNAName[b][1])
    pd.DataFrame(milist).to_csv(r'milist.csv',index=None,header=None)
    print(len(milist))

    label = MyLabel(data)
    datadat = pd.DataFrame({'miRNA':milist,'drug':druglist,'label':label}).to_csv(r'datadat.csv',index=None,header=None)


