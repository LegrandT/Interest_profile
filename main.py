import xlrd
import pandas as pd
from sklearn import datasets
from math import *
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import operator
import random
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 100)


# file_location1 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Opens.csv"
# odfOpen = pd.read_csv(file_location1)
# file_location2 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Sends.csv"
# odfSends = pd.read_csv(file_location2)
# file_location3 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Clicks.csv"
# odfClicks = pd.read_csv(file_location3, encoding= 'ISO-8859-1')
# file_location4 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\Customer_data.csv"
# odfCosDat = pd.read_csv(file_location4)
# file_location5 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\newsletter_link_tags_def.csv"
# odfNewLiTaDe = pd.read_csv(file_location5)
# file_location6 = "C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\newsletter_link_tags_0817.csv"
# odfNewLiTa08 = pd.read_csv(file_location6)
file_location7 = 'C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\PreprocessedData1000.xlsx'
odfAll = pd.read_excel(file_location7)



def getLabel(train):
    labels = []
    for i in range(len(train)):
        if train[i] not in labels:
            labels.append(train[i])
    return labels

#calculate euclidean distance
def eucliDist(a, b):
    d = 0
    for i in range(len(a)):
        d += pow((a[i] - b[i]), 2)
    return sqrt(d)


def muCl(x_training, ytraining, x, knear):
    # nearest = {}
    # for i in range(len(x_training)):
    #     nearest[i] = eucliDist(x_training[i], x)
    # nearest = sorted(nearest.items(), key=operator.itemgetter(1))
    #
    # classes = getLabel(ytraining)
    #
    #
    # k_count = [0]*len(classes)
    # for i in range(knear):
    #     for j in range(len(classes)):
    #         if ytraining[nearest[i][0]] == classes[j]:
    #             k_count[j] += 1
    #
    # result = classes[k_count.index(max(k_count))]
    neigh = KNeighborsClassifier(n_neighbors=knear)
    neigh.fit(x_training, ytraining)
    result = neigh.predict([x])
    return result

def Accuracy(predict, accur): # using 0/1 loss
    error = 0
    errors = []
    for i in range(len(predict)):
        if predict[i] != accur.iloc[i,:].tolist():
            error +=1
            # errors.append([predict[i],"should be", accur.iloc[i,:].tolist()]
    # print(error, "errors are", errors)
    return 100*(1 - error / len(predict))

def FScore(predict, accur): # using 0/1 loss
    tp = 0
    fp= 0
    fn = 0
    for i in range(len(predict)):
        if predict[i] == 1:
            if accur[i] ==1:
                tp+=1
            else:
                fp+=1
        elif predict[i] == 0:
            if accur[i] == 1:
                fn +=1

    preci = tp/(tp+fp)
    recall = tp/(tp+fn)
    return 100*2*((preci*recall)/(preci + recall))

df = odfAll.copy()
# display(df.shape)
# sampleSize = 10000
#
# idxdf = np.random.permutation(df.index)
# df = df.reindex(idxdf)
# df = df.reset_index(drop=True)
#
# df = df[0:sampleSize+1]
#
# writer = pd.ExcelWriter('C:\\Users\\xg16137\\OneDrive - APG\\My Documents\\Projects Weiwei\\Interest profile\\PreprocessedData1000.xlsx')
# df.to_excel(writer, 'Sheet1')
# writer.save()

display(df)
print(odfAll)
x = []
y = []

x = df.drop(['Unnamed: 0', 'Unnamed: 0.1', 'KLTID', 'GIFSF', 'MD', 'OB', 'DZ', 'DZ', 'PK', 'sum', 'fGIFSF', 'fMD', 'fOB', 'fDZ', 'fDZ', 'fPK'], axis=1)

# display(x)

y = df[['GIFSF', 'MD', 'OB', 'DZ', 'PK']].copy()

# display(y)

BRave = []
LPave = []
rakelAve = []
kmeanAve = []

# define the range for the k in knn
firstK = 10
lastK = 20
krange = range(firstK, lastK + 1)


oneK = True # if we already chosed a unique k

if firstK != lastK:
    oneK = False
    for i in range(len(krange)):
        BRave.append([])
        LPave.append([])
        rakelAve.append([])
        kmeanAve.append([])

iterations = 2 # number of iterations


for r in range(iterations):
    print("iteration ", r+1, '/', iterations)
    # shuffle the data to have homogenized data (avoid having whole folder of same class)

    idx = np.random.permutation(x.index)
    x = x.reindex(idx)
    y = y.reindex(idx)
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)


 # link different k for the knn
    for k in krange:
        print("k =",k , '/', range(firstK, lastK))

        # data divided in 10 folder for cross validation
        kfolder = 10
        #
        # to register the accuracy for each iteration of cross-validation
        BRacc = []
        LPacc = []
        rakelacc = []
        Kmeanacc = []

        # for each folder to be taken as link data
        for p in range(kfolder):
            print("cross-validation", p+1, '/', kfolder)

            # separate folders between test and train data
            xtest = x[(x.index >= p*(len(x.index) / kfolder)) & (x.index < (p + 1)*(len(x.index) / kfolder))]
            ytest = y[(y.index >= p*(len(y.index) / kfolder)) & (y.index < (p + 1)*(len(y.index) / kfolder))]
            xtrain = pd.concat([x[(x.index < p*(len(x.index) / kfolder))], x[(x.index >= (p + 1)*(len(x.index) / kfolder))]])
            ytrain = pd.concat([y[(y.index < p*(len(y.index) / kfolder))], y[(y.index >= (p + 1)*(len(y.index) / kfolder))]])

            # sm = SMOTENC(random_state=12, categorical_features=[18, 19])



            # Binary relevance
            #  can create one more label, when classification
            BRresult = []
            for i in range(len(xtest)):
                BRresult.append([])

            sast = [1/2, 1/2, 1, 1/2, 1/2] # ratio for smote
            for i in range(len(ytrain.columns)):
                # smote for data balance
                # print(ytrain.iloc[:,i].value_counts())
                sm = SMOTE(random_state=42, sampling_strategy=sast[i])
                xBR, yBR = sm.fit_sample(xtrain, ytrain.iloc[:,i])

                xBR = pd.DataFrame(data=xBR, columns=xtrain.columns)
                # print(list(yBR).count(0), list(yBR).count(1))
                # random forest feature selection
                rfc = RandomForestClassifier(n_estimators=100)
                rfc.fit(xBR, yBR)
                # print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), xtrain.columns), reverse=True))
                plt.plot(xtrain.columns, rfc.feature_importances_, 'b', label='Binary relevance')
                plt.show()
                sel = SelectFromModel(rfc, threshold=0.15)
                sel.fit(xBR, yBR)
                selected_feat = list(xBR.columns[(sel.get_support())])

                # Lasso regression for feature selection
                lasso = Lasso(alpha=0.0001, max_iter=10e5)
                lasso.fit(xBR, yBR)
                train_score = lasso.score(xtest, ytest.iloc[:,i])
                test_score = lasso.score(xtest, ytest.iloc[:,i])
                coeff_used = np.sum(lasso.coef_ != 0)
                print("training score:", train_score)
                print("test score: ", test_score)
                print("number of features used: ", coeff_used)

                # print(selected_feat)
                xBR = xBR[selected_feat]

                neigh = KNeighborsClassifier(n_neighbors=k)
                neigh.fit(xBR, yBR)

                for j in range(len(xtest)):
                    # print(j, 'out of ', len(xtest))
                    BRresult[j].append(neigh.predict([xtest[selected_feat].iloc[j,:]])[0])

            # fscore = FScore([row[1] for row in BRresult], list(ytest.iloc[:,1]))
            # print(fscore)
            accur = Accuracy(BRresult, ytest)
            print("BR", accur)
            BRacc.append(accur)


            # Label powerset
            #
            # temp = ytrain.values.tolist()
            # yLPtemp2 = [''.join(str(s) for s in u) for u in temp]
            # yLPtemp = pd.DataFrame()
            # yLPtemp['combined'] = yLPtemp2
            # # print(yLPtemp['combined'].value_counts())
            # sm = SMOTE('not majority')
            # xLP, yLP = sm.fit_sample(xtrain, yLPtemp)
            # xLP = pd.DataFrame(data=xLP, columns=xtrain.columns)

            # sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
            # sel.fit(xLP, yLP)
            # sel.get_support()
            # selected_feat = list(xLP.columns[(sel.get_support())])

            # xLP = xLP[selected_feat]

            # LPresult = []
            # neigh = KNeighborsClassifier(n_neighbors=k)
            # neigh.fit(xLP, yLP)
            # for i in range(len(xtest)):
            #     LPresult.append(neigh.predict([xtest.iloc[i,:]])[0])
            # LPresult = [list(map(int,list(u))) for u in LPresult]
            # accur = Accuracy(LPresult, ytest)
            # print("LP", accur)
            # LPacc.append(accur)



        if oneK:
            BRave.append(sum(BRacc) / len(BRacc))
            # LPave.append(sum(LPacc) / len(LPacc))
            # rakelAve.append(sum(rakelacc) / len(rakelacc))
            # kmeanAve.append(sum(Kmeanacc) / len(Kmeanacc))
        else:
            BRave[k - firstK].append(sum(BRacc) / len(BRacc))
            # LPave[k - firstK].append(sum(LPacc) / len(LPacc))
            # rakelAve[k - firstK].append(sum(rakelacc) / len(rakelacc))
            # kmeanAve[k - firstK].append(sum(Kmeanacc) / len(Kmeanacc))

if oneK :
    BRave = sum(BRave) / len(BRave)
    # LPave = sum(LPave) / len(LPave)
    # rakelAve = sum(rakelAve) / len(rakelAve)
    # kmeanAve = sum(kmeanAve) / len(kmeanAve)
    # print('K-mean clutering accuracy is', kmeanAve, '%')
    print('Binary relevance accuracy is', BRave, '%')
    # print('Label powerset accuracy is', LPave, '%')
    # print('Rakel result accuracy is', rakelAve, '%')
else:
    for r in range(len(krange)):
        BRave[r] = sum(BRave[r]) / len(BRave[r])
        # LPave[r] = sum(LPave[r]) / len(LPave[r])
        # rakelAve[r] = sum(rakelAve[r]) / len(rakelAve[r])
        # kmeanAve[r] = sum(kmeanAve[r]) / len(kmeanAve[r])
    xaxe = []
    for i in krange:
        xaxe.append(i)
    plt.plot(xaxe, BRave, 'b', label='Binary relevance')
    # plt.plot(xaxe, LPave, 'g', label='Label powerset')
    # plt.plot(xaxe, rakelAve, 'r', label='rakel')
    plt.legend()
    plt.show()