import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau
from keras.utils import np_utils
import csv
import pandas as pd
import numpy as np

csv.field_size_limit(500 * 1024 * 1024)

def GenerateBehaviorFeature(InteractionPair, NodeBehavior):
    SampleFeature1 = []
    SampleFeature2 = []
    #miss = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][0])#miRNA
        Pair2 = str(InteractionPair[i][1])#drug

        for m in range(len(NodeBehavior)):
            if Pair1 == NodeBehavior[m][0]:#miRNA
                SampleFeature1.append(NodeBehavior[m][1:])
                break
            #SampleFeature1.append(miss)
        for n in range(len(NodeBehavior)):
            if Pair2 == NodeBehavior[n][0]:#drug
                SampleFeature2.append(NodeBehavior[n][1:])
                break
            #SampleFeature2.append(miss)
    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')
    return SampleFeature1, SampleFeature2

def GenerateAttributeFeature(InteractionPair, drug_feature, miRNA_feature):
    SampleFeature1 = []
    SampleFeature2 = []
    for i in range(len(InteractionPair)):
        Pair1 = str(InteractionPair[i][1])  #drug
        Pair2 = str(InteractionPair[i][0])  #mirna
        for m in range(len(drug_feature)):#drug
            if int(Pair1) == int(drug_feature[m][0]):
                SampleFeature1.append(drug_feature[m][1:])
                break
        for n in range(len(miRNA_feature)):#mirna
            if Pair2 == str(miRNA_feature[n][0]):
                SampleFeature2.append(miRNA_feature[n][1:])
                break

    SampleFeature1 = np.array(SampleFeature1).astype('float32')
    SampleFeature2 = np.array(SampleFeature2).astype('float32')

    return SampleFeature1, SampleFeature2

def MyLabel(Sample):
    label = []
    for i in range(int(len(Sample) / 2)):
        label.append(1)
    for i in range(int(len(Sample) / 2)):
        label.append(0)
    return label


if __name__ == '__main__':

    dataset_list = ['ncDR','RNAInter','SM2miR1','SM2miR2','SM2miR3']

for dataset in dataset_list:
    AllNodeBehavior = pd.read_csv(r'./Feature_topological/'+dataset+' BiNE 64.csv', header=None).values.tolist()
    AllDrugFingerPrint = pd.read_csv(r'Feature_attribute/'+dataset+' drug MACCS.csv', header=None).astype(np.float).values.tolist()
    AllMiRNAKer = pd.read_csv(r'Feature_attribute/'+dataset+' miRNA kmer.csv', header=None).values.tolist()

    PositiveSample_Train = pd.read_csv(dataset+'/PositiveSample_Train.csv', header=None).values.tolist()
    PositiveSample_Validation = pd.read_csv(dataset+'/PositiveSample_Validation.csv', header=None).values.tolist()
    PositiveSample_Test = pd.read_csv(dataset+'/PositiveSample_Test.csv', header=None).values.tolist()

    NegativeSample_Train = pd.read_csv(dataset+'/NegativeSample_Train.csv', header=None).values.tolist()
    NegativeSample_Validation = pd.read_csv(dataset+'/NegativeSample_Validation.csv', header=None).values.tolist()
    NegativeSample_Test = pd.read_csv(dataset+'/NegativeSample_Test.csv', header=None).values.tolist()

    x_train_pair = []
    x_train_pair.extend(PositiveSample_Train)
    x_train_pair.extend(NegativeSample_Train)

    x_validation_pair = []
    x_validation_pair.extend(PositiveSample_Validation)
    x_validation_pair.extend(NegativeSample_Validation)

    x_test_pair = []
    x_test_pair.extend(PositiveSample_Test)
    x_test_pair.extend(NegativeSample_Test)

    x_train_1_Attribute, x_train_2_Attribute = GenerateAttributeFeature(x_train_pair, AllDrugFingerPrint,AllMiRNAKer)
    x_validation_1_Attribute, x_validation_2_Attribute = GenerateAttributeFeature(x_validation_pair,AllDrugFingerPrint,AllMiRNAKer)
    x_test_1_Attribute, x_test_2_Attribute = GenerateAttributeFeature(x_test_pair, AllDrugFingerPrint,AllMiRNAKer)

    x_train_1_Behavior, x_train_2_Behavior = GenerateBehaviorFeature(x_train_pair, AllNodeBehavior)
    x_validation_1_Behavior, x_validation_2_Behavior = GenerateBehaviorFeature(x_validation_pair, AllNodeBehavior)
    x_test_1_Behavior, x_test_2_Behavior = GenerateBehaviorFeature(x_test_pair, AllNodeBehavior)

    num_classes = 2
    y_train_Pre = MyLabel(x_train_pair)     # Label->one hot
    y_validation_Pre = MyLabel(x_validation_pair)
    y_test_Pre = MyLabel(x_test_pair)

    y_train = np_utils.to_categorical(y_train_Pre, num_classes)
    y_validation = np_utils.to_categorical(y_validation_Pre, num_classes)
    y_test = np_utils.to_categorical(y_test_Pre, num_classes)

    print('x_train_1_Attribute shape', x_train_1_Attribute.shape)
    print('x_train_2_Attribute shape', x_train_2_Attribute.shape)
    print('x_train_1_Behavior shape', x_train_1_Behavior.shape)
    print('x_train_2_Behavior shape', x_train_2_Behavior.shape)

    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    input1 = Input(shape=(len(x_train_1_Attribute[0]),), name='input1')
    x1 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.05))(input1)
    x1 = Dropout(rate=0.3)(x1)
    input2 = Input(shape=(len(x_train_2_Attribute[0]),), name='input2')
    x2 = Dense(64, activation='relu', activity_regularizer=regularizers.l2(0.05))(input2)
    x2 = Dropout(rate=0.3)(x2)
    input3 = Input(shape=(len(x_train_1_Behavior[0]),), name='input3')
    x3 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.05))(input3)
    x3 = Dropout(rate=0.3)(x3)
    input4 = Input(shape=(len(x_train_2_Behavior[0]),), name='input4')
    x4 = Dense(128, activation='relu', activity_regularizer=regularizers.l2(0.05))(input4)
    x4 = Dropout(rate=0.3)(x4)
    flatten = keras.layers.concatenate([x1, x2, x3, x4])
    hidden = Dense(32, activation='relu', name='hidden2', activity_regularizer=regularizers.l2(0.01))(flatten)
    hidden = Dropout(rate=0.3)(hidden)
    output = Dense(num_classes, activation='softmax', name='output')(hidden)  # category
    model = Model(inputs=[input1, input2, input3, input4], outputs=output)
    model.summary()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='auto')  # Automatically adjust the learning rate
    history = model.fit({'input1': x_train_1_Attribute, 'input2': x_train_2_Attribute, 'input3': x_train_1_Behavior, 'input4': x_train_2_Behavior}, y_train,
                        validation_data=({'input1': x_validation_1_Attribute, 'input2': x_validation_2_Attribute,
                                          'input3': x_validation_1_Behavior, 'input4': x_validation_2_Behavior}, y_validation),
                        callbacks=[reduce_lr],
                        epochs=60, batch_size=128,   #epochs=50, batch_size=128,
                        )

    # StorFile(MyChange(history.history['val_loss']), 'Val_Loss.csv')
    # StorFile(MyChange(history.history['val_accuracy']), 'Val_ACC.csv')
    # StporFile(MyChange(history.history['loss']), 'Loss.csv')
    # StorFile(MyChange(history.history['acc']), 'ACC.csv')
    model.save(dataset+'model.h5')

    ModelTest = Model(inputs=model.input, outputs=model.get_layer('output').output)
    ModelTestOutput = ModelTest.predict([x_test_1_Attribute, x_test_2_Attribute, x_test_1_Behavior, x_test_2_Behavior])

    print(ModelTestOutput.shape)
    print(type(ModelTestOutput))


    LabelPredictionProb = []
    LabelPrediction = []

    counter = 0
    while counter < len(ModelTestOutput):
        rowProb = []
        rowProb.append(y_test_Pre[counter])
        rowProb.append(ModelTestOutput[counter][1])
        LabelPredictionProb.append(rowProb)

        row = []
        row.append(y_test_Pre[counter])
        if ModelTestOutput[counter][1] > 0.5:
            row.append(1)
        else:
            row.append(0)
        LabelPrediction.append(row)

        counter = counter + 1
    pd.DataFrame(LabelPredictionProb).to_csv(dataset+'/Prediction Probability.csv')
    pd.DataFrame(LabelPredictionProb).to_csv(dataset+'/Prediction resutl.csv')
