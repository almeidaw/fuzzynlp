from pycorenlp import StanfordCoreNLP
from senticnet.senticnet import SenticNet
# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import os
import pathlib
import pysentiment2 as ps

# Starts the SenticNet and Stanford Core NLP tools
sn = SenticNet()
nlp = StanfordCoreNLP('http://localhost:9000')
#
domains = {}
sumZjc = {}
# Creates a Pandas dataframe
domain_df = pd.DataFrame
# Defines the training set directory
trainingDir = pathlib.Path(__file__).parent.absolute().joinpath('dataset-teste2')

# Access every item inside the training set directory
for dir in os.listdir(trainingDir):
    # Defines lists to temporarily store the dataframe columns
    features = []
    pos = []
    peic = []
    kic = []
    sic = []
    zic = []
    ni = 0
    ni_list = []
    difreq = []
    #
    dirPath = os.path.join(trainingDir, dir)

    # Check if the accessed item is a directory and it's not hidden
    if os.path.isdir(dirPath) and dir[0] != ".":
        files = os.listdir(dirPath)

        # Access every not hidden file inside the domain folder
        for file in files:
            if file[0] != ".":
                # Just to check which file is being read at the time
                print(file)

                # Check polarity expressed in the file name (which is the same polarity as the documents it stores)
                # If "pos" is in the name the polarity is 1, otherwise it's -1
                if "pos" in file:
                    polarity = 1
                else:
                    polarity = -1

                #try here
                # Opens file and reads next line while it's not empty. Each line is a document
                with open("%s/%s" % (dirPath, file)) as dataset:
                    line = dataset.readline()

                    while line:
                        #
                        inDocument = False
                        # Get Stanford Core NLP lemma, pos and dependency tree outputs for that document
                        coreOutput = nlp.annotate(line,
                                                  properties={'annotators': 'lemma, pos, depparse',
                                                              'outputFormat': 'json', 'timeout': 100000})
                        # Get each token in each sentence of the document and check if
                        # their pos is noun, adjective, verb or adverb
                        for sentence in coreOutput["sentences"]:
                            for token in sentence["tokens"]:
                                if token["pos"] in ["NN", "NNS", "NNP", "NNPS", "VB", "VBD", "VBG", "VBN", "VBP",
                                                    "VBZ", "RB", "RBR", "RBS", "JJS", "JJR", "JJ"]:

                                    # Get the lemma of the token, which is what we're going to use
                                    lemma = token["lemma"]

                                    # If the lemma is not in the feature list yet we add it to the list and
                                    # also add entries for it in the lists for polarity and occurrences counting
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

        # Ni is the sum of all occurrences of all features (sum of all Zs)
        # so it's computed after we have passed for all the files in the domain
        ni = sum(zic)

        # Get the index of each item in the feature column of the dataframe and compute the other values
        # that also depends on values for all the domain as SiC ans Ni
        for feature in features:
            index = features.index(feature)
            peic.insert(index, kic[index] / sic[index])
            difreq.insert(index, zic[index]/ni)
            ni_list.insert(index, ni)

        # Writes the lists as columns of the dataframe
        domain_df = pd.DataFrame(list(zip(features, pos, kic, sic, peic, zic, ni_list, difreq)), columns=[
            "FEATURE", "POS", "SUM OF POLARITIES (K)", "# OF DOCS WITH FEATURE (S)",
            "ESTIMATED POLARITY (P=K/S)", "# TIMES OF FEAT. IN DOMAIN (Z)",
            "# TIMES OF ALL FEAT. IN DOMAIN (N)", "RELEVANCE OF FEAT. IN DOMAIN (FREQ=Z/N)"])
        # Saves the dataframe as an entry in a dictionary
        domains.update({dir: domain_df})

 #  dict in HIV4 is preprocessed by the default tokenizer in Library


for domain, dataframe in domains.items():
    zjc = []
    diuniq = []
    DBDi = []
    pcs = []
    pcg = []

    for index, row in dataframe.iterrows():
        zic = row["# TIMES OF FEAT. IN DOMAIN (Z)"]
        difreq = row["RELEVANCE OF FEAT. IN DOMAIN (FREQ=Z/N)"]
        zjc.insert(index, sumZjc.get(row["FEATURE"]))
        diuniq.insert(index, zic/zjc[index])
        DBDi.insert(index, difreq*diuniq[index])
        pcg.insert(index, list(ps.HIV4().get_score(ps.HIV4().tokenize(row["FEATURE"])).values())[2])

        try:
            pcs.insert(index, sn.polarity_intense(row["FEATURE"]))
        except:
            pcs.insert(index, "")

    # Adds the rest of the columns to dataframe
    dataframe["# TIMES OF FEAT. IN ALL DOMAINS (SUM_Z)"] = zjc
    dataframe["RELEVANCE OF FEAT. IN DOMAIN (UNIQ=Z/SUM_Z)"] = diuniq
    dataframe["DOMAIN BELONGING DEGREE (DBD)"] = DBDi
    dataframe["GENERAL INQUIRER"] = pcg
    dataframe["SENTICNET"] = pcs


    # Finally writes dataframe to a CSV file
    try:
        dataframe.to_csv("outputs/%s.csv" % domain)
    except:
        print("Unable to write CSV file")
    else:
        print("CSV file written successfully")

