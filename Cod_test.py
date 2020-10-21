from pycorenlp import StanfordCoreNLP
#from senticnet.senticnet import SenticNet
#import matplotlib.pyplot as plt
#import numpy as np
import pandas as pd
import os
import pathlib

#sn = SenticNet()
nlp = StanfordCoreNLP('http://localhost:9000')
domains = {}
sumZjc = {}
domain_df = pd.DataFrame
trainingDir = pathlib.Path(__file__).parent.absolute().joinpath('dataset-teste')


for dir in os.listdir(trainingDir):
    features = []
    pos = []
    peic = []
    kic = []
    sic = []
    zic = []
    ni = 0
    difreq = []
    dirPath = os.path.join(trainingDir, dir)

    if os.path.isdir(dirPath) and dir[0] != ".":
        files = os.listdir(dirPath)

        for file in files:
            print(file)
            if "pos" in file:
                polarity = 1
            else:
                polarity = -1

            with open("%s/%s" % (dirPath, file)) as dataset:
                line = dataset.readline()

                while line:
                    inDocument = False
                    coreOutput = nlp.annotate(line,
                                              properties={'annotators': 'lemma, pos, depparse', 'outputFormat': 'json',
                                                          'timeout': 100000})
                    for sentence in coreOutput["sentences"]:
                        for token in sentence["tokens"]:
                            if token["pos"] in ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
                                                "VBZ", "RB", "RBR", "RBS", "JJS", "JJR", "JJ"]:

                                lemma = token["lemma"]

                                if not(lemma in features):
                                    features.append(lemma)
                                    kic.append(polarity)
                                    sic.append(1)
                                    zic.append(1)
                                    pos.append(token["pos"])
                                    inDocument = True

                                else:
                                    index = features.index(lemma)
                                    zic[index] += 1
                                    if not inDocument:
                                        kic[index] += polarity
                                        sic[index] += 1
                                        inDocument = True

                                if not(lemma in sumZjc):
                                    sumZjc.update({lemma: 1})
                                else:
                                    sumZjc.update({lemma: (sumZjc.get(lemma) + 1)})

                    line = dataset.readline()

        ni = sum(zic)

        for feature in features:
            index = features.index(feature)
            peic.append(kic[index] / sic[index])
            difreq.append(zic[index]/ni)

        domain_df = pd.DataFrame(list(zip(features, pos, kic, sic, peic, zic, difreq)), columns=[
            "Feature", "POS", "KiC", "SiC", "PeiC", "ZiC", "Di_Freq"])
        domains.update({dir: domain_df})

for domain, dataframe in domains.items():
    zjc = []
    diuniq = []
    DBDi = []
    pcs = []
    for index, row in dataframe.iterrows():
        zic = row["ZiC"]
        difreq = row["Di_Freq"]
        zjc.insert(index, sumZjc.get(row["Feature"]))
        diuniq.insert(index, zic/zjc[index])
        DBDi.insert(index, difreq*diuniq[index])
        # pcs.insert(index, sn.polarity_intense(row["Feature"]))

    dataframe["Sum_Zjc"] = zjc
    dataframe["Di_Uniq"] = diuniq
    dataframe["DBDi"] = DBDi
    # dataframe["SenticNet"] = pcs
    dataframe.to_csv("outputs/%s.csv" % domain)
