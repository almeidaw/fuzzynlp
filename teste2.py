from pycorenlp import StanfordCoreNLP
from senticnet.senticnet import SenticNet
from statistics import variance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
import pysentiment2 as ps
import time

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

def core_nlp():
    processed_domains = []
    # Access every item inside the training set directory
    for dir in os.listdir(trainingDir):

        for domain in os.listdir('outputs2'):
            if domain[0] != "." and domain not in processed_domains:
                processed_domains.append(domain[:-4])

        if dir in processed_domains:
            domain_df = pd.read_csv("outputs2/%s.csv" % dir)
            domains.update({dir: domain_df})
            print("Domain", dir, "had already been processed. Loaded it to Datagram.")
            continue

        with open('outputs2/'+dir+'.csv', 'w') as fp:
            pass
        print("Processing files for", dir)
        start2 = time.time()

        # Defines lists to temporarily store the dataframe columns
        lemmas = []
        tokens = []
        type_feat = []
        dependencies = []
        pos = []
        peic = []
        kic = []
        sic = []
        zic = []
        ni = 0
        ni_list = []
        difreq = []

        dirPath = os.path.join(trainingDir, dir)
        # Check if the accessed item is a directory and it's not hidden
        if os.path.isdir(dirPath) and dir[0] != ".":
            files = os.listdir(dirPath)
            # Access every not hidden file inside the domain folder
            for file in files:
                if file[0] != ".":
                    # Just to check which file is being read at the time
                    start = time.time()

                    # Check polarity expressed in the file name (which is the same polarity as the documents it stores)
                    # If "pos" is in the name the polarity is 1, otherwise it's -1
                    if "pos" in file:
                        polarity = 1
                    else:
                        polarity = -1

                    # try here
                    # Opens file and reads next line while it's not empty. Each line is a document
                    with open("%s/%s" % (dirPath, file)) as dataset:
                        line = dataset.readline()

                        while line:
                            #
                            in_document = set()
                            # Get Stanford Core NLP lemma, pos and dependency tree outputs for that document
                            core_output = nlp.annotate(line,
                                                      properties={'annotators': 'lemma, pos, depparse',
                                                                  'outputFormat': 'json', 'timeout': 100000})
                            # Get each token in each sentence of the document and check if
                            # their pos is noun, adjective, verb or adverb
                            for sentence in core_output["sentences"]:
                                # Gets complex features
                                for dependency in sentence["basicDependencies"]:
                                    # Defines list of possible dependencies
                                    nouns = ("NN", "NNS", "NNP", "NNPS")
                                    adjectives = ("JJS", "JJR", "JJ")
                                    verbs = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
                                    adverbs = ("RB", "RBR", "RBS")

                                    # We dont want punctuation or root dependencies
                                    if not dependency["dep"] == "ROOT" and not dependency["dep"] == "punct":
                                        # Checks if its a match
                                        dependent = sentence["tokens"][dependency["dependent"]-1]
                                        governor = sentence["tokens"][dependency["governor"]-1]
                                        pos_dep = dependent["pos"]
                                        pos_gov = governor["pos"]
                                        if (pos_dep in nouns and (pos_gov in adjectives or pos_gov in verbs)) \
                                                or (pos_dep in adjectives and (pos_gov in nouns or pos_gov in verbs or pos_gov in adverbs)) \
                                                or (pos_dep in verbs and (pos_gov in nouns or pos_gov in adjectives)) \
                                                or (pos_dep in adverbs and pos_gov in adjectives) \
                                                or (pos_gov in nouns and (pos_dep in adjectives or pos_dep in verbs))\
                                                or (pos_gov in adjectives and (pos_dep in nouns or pos_dep in verbs or pos_dep in adverbs))\
                                                or (pos_gov in verbs and (pos_dep in nouns or pos_dep in adjectives)) \
                                                or (pos_gov in adverbs and pos_dep in adjectives):
                                            lemma = governor["lemma"]+"-"+dependent["lemma"]
                                            lemma2 = dependent["lemma"]+"-"+governor["lemma"]
                                            token = governor["word"]+"-"+dependent["word"]

                                            if lemma not in lemmas and lemma2 not in lemmas:
                                                lemmas.append(lemma)
                                                tokens.append(token)
                                                dependencies.append(dependency["dep"])
                                                type_feat.append("complex")
                                                kic.append(polarity)
                                                sic.append(1)
                                                zic.append(1)
                                                pos.append(pos_gov+"-"+pos_dep)
                                                in_document.add(lemma)
                                                in_document.add(lemma2)

                                            else:
                                                try:
                                                    index_lemma = lemmas.index(lemma)
                                                except:
                                                    index_lemma = lemmas.index(lemma2)
                                                zic[index_lemma] += 1
                                                if lemma not in in_document:
                                                    kic[index_lemma] += polarity
                                                    sic[index_lemma] += 1
                                                    in_document.add(lemma)
                                                    in_document.add(lemma2)

                                            if lemma not in sumZjc and lemma not in sumZjc:
                                                sumZjc.update({lemma: 1})
                                            else:
                                                try:
                                                    sumZjc.update({lemma: (sumZjc.get(lemma) + 1)})
                                                except:
                                                    sumZjc.update({lemma2: (sumZjc.get(lemma2) + 1)})

                                # Gets simple features
                                for token in sentence["tokens"]:
                                    if token["pos"] in nouns or token["pos"] in adjectives or \
                                            token["pos"] in verbs or token["pos"] in adverbs:

                                        # Get the lemma of the token, which is what we're going to use
                                        lemma = token["lemma"]

                                        # If the lemma is not in the feature list yet we add it to the list and
                                        # also add entries for it in the lists for polarity and occurrences counting
                                        if not (lemma in lemmas):
                                            lemmas.append(lemma)
                                            tokens.append(token["word"])
                                            dependencies.append("-")
                                            type_feat.append("simple")
                                            kic.append(polarity)
                                            sic.append(1)
                                            zic.append(1)
                                            pos.append(token["pos"])
                                            in_document.add(lemma)

                                        else:
                                            index_lemma = lemmas.index(lemma)
                                            zic[index_lemma] += 1
                                            if lemma not in in_document:
                                                kic[index_lemma] += polarity
                                                sic[index_lemma] += 1
                                                in_document.add(lemma)

                                        if lemma not in sumZjc:
                                            sumZjc.update({lemma: 1})
                                        else:
                                            sumZjc.update({lemma: (sumZjc.get(lemma) + 1)})

                            line = dataset.readline()

                    end = time.time()
                    print("Processed", file, "in", (end-start)/60)

            start = time.time()
            print("Processing additional information for each feature in domain", dir)
            # Ni is the sum of all occurrences of all features (sum of all Zs)
            # so it's computed after we have passed for all the files in the domain
            ni = sum(zic)

            # Get the index of each item in the feature column of the dataframe and compute the other values
            # that also depends on values for all the domain as SiC and Ni
            for lemma in lemmas:
                index_lemma = lemmas.index(lemma)
                peic.insert(index_lemma, kic[index_lemma] / sic[index_lemma])
                difreq.insert(index_lemma, zic[index_lemma] / ni)
                ni_list.insert(index_lemma, ni)

            # Writes the lists as columns of the dataframe
            domain_df = pd.DataFrame(list(zip(tokens, lemmas, type_feat, dependencies, pos, kic, sic, peic, zic, ni_list, difreq)), columns=[
                "RAW FEATURE", "FEATURE", "TYPE", "DEPENDENCY", "POS", "SUM OF POLARITIES (K)", "# OF DOCS W/ FEATURE (S)",
                "ESTIMATED POLARITY (P=K/S)", "# TIMES OF FEAT. IN DOMAIN (Z)",
                "# TIMES OF ALL FEAT. IN DOMAIN (N)", "RELEVANCE OF FEAT. IN DOMAIN (FREQ=Z/N)"])

            end = time.time()
            print("Processed additional information for each feature in domain", dir, "in", (end-start)/60)
            print("Processed domain", dir, "in", (end-start2)/60)

            # Finally writes dataframe to a CSV file
            try:
                domain_df.to_csv("outputs2/%s.csv" % dir)
            except:
                print("Unable to write CSV file for %s" % dir)
            else:
                print("CSV file for %s has been successfully written" % dir)

            # Saves the dataframe as an entry in a dictionary
            domains.update({dir: domain_df})


def dbd():
    print("\n---DBD step---\n")
    for domain, dataframe in domains.items():
        print("Calculating DBD for domain", domain)
        zjc = []
        diuniq = []
        DBDi = []

        for index, row in dataframe.iterrows():
            zic = row["# TIMES OF FEAT. IN DOMAIN (Z)"]
            difreq = row["RELEVANCE OF FEAT. IN DOMAIN (FREQ=Z/N)"]
            zjc.insert(index, sumZjc.get(row["FEATURE"]))
            diuniq.insert(index, zic/zjc[index])
            DBDi.insert(index, difreq*diuniq[index])

        # Adds the rest of the columns to dataframe
        dataframe["# TIMES OF FEAT. IN ALL DOMAINS (SUM_Z)"] = zjc
        dataframe["RELEVANCE OF FEAT. IN DOMAIN (UNIQ=Z/SUM_Z)"] = diuniq
        dataframe["DOMAIN BELONGING DEGREE (DBD=FREQ*UNIQ)"] = DBDi

        # Updates dataframe to a CSV file
        try:
            dataframe.to_csv("outputs2/%s.csv" % domain)
        except:
            print("Unable to update CSV file for domain %s" % domain)
        else:
            print("CSV file for domain %s has been successfully updated" % domain)

def trapezoid(a,b,c,d):
    eixoY = [0, 1, 1, 0]


    plt.plot(a,b,c,d, eixoY , 'go')  # green bolinha
    plt.plot(a,b,c,d, eixoY , 'k:', color='orange')  # linha pontilha orange

    plt.title("Trapezoid Teste")

    plt.grid(True)
    plt.xlabel("eixo horizontal")
    plt.ylabel("eixo y")
    plt.show()

def refinamento():
    print("\n---Refinement step---\n")

    for domain, dataframe in domains.items():
        print("Refining results for domain", domain)
        pcs = []
        pcg = []
        avg = []
        var = []
        a =[]
        b= []
        c =[]
        d=[]

        for index, row in dataframe.iterrows():
            if row["TYPE"] == "complex":
                try:
                    feature1, feature2 = row["FEATURE"].split("-")
                except:
                   feature1 = row["FEATURE"]
                   feature2 = row["FEATURE"]
                # get PCG values

                feature1_pcg = list(ps.HIV4().get_score(ps.HIV4().tokenize(feature1)).values())[2]

                feature2_pcg = list(ps.HIV4().get_score(ps.HIV4().tokenize(feature2)).values())[2]

                if row["DEPENDENCY"] == "advmod":
                    pcg.insert(index, float(feature1_pcg)*float(feature2_pcg))
                else:
                    pcg.insert(index, (float(feature1_pcg)+float(feature2_pcg))/2)
            else:
                pcg.insert(index, list(ps.HIV4().get_score(ps.HIV4().tokenize(row["FEATURE"])).values())[2])
            try:
                pcs.insert(index, sn.polarity_intense(row["FEATURE"]))
            except:
                pcs.insert(index, 0)
            #média e variance
            avg.insert(index,(float(pcs[index])+float(pcg[index])+float(row["ESTIMATED POLARITY (P=K/S)"]))/3)
            lista=[float(pcs[index]),float(pcg[index]),float(row["ESTIMATED POLARITY (P=K/S)"])]
            var.insert(index, variance(lista))
            # the fuzzy membership function is transformed into a trapezoid with the vertexes(a, b, c,d)
            b.insert(index, min(float(row["ESTIMATED POLARITY (P=K/S)"]), float(avg[index])))
            a.insert(index, max(-1, (float(b[index]) - float(var[index]))))
            c.insert(index, max(float(row["ESTIMATED POLARITY (P=K/S)"]), float(avg[index])))
            d.insert(index, min(1, float(c[index]) + float(var[index])))
            #trapezoid(a[index], b[index], c[index], d[index])


        dataframe["SENTICNET"] = pcs
        dataframe["GENERAL INQUIRER"] = pcg
        dataframe["AVERAGE"] = avg
        dataframe["VARIANCE"] = var
        dataframe["A"] = a
        dataframe["B"] = b
        dataframe["C"] = c
        dataframe["D"] = d

        # Updates dataframe to a CSV file
        try:
            dataframe.to_csv("outputs2/%s.csv" % domain)
        except:
            print("Unable to update CSV file for domain %s" % domain)
        else:
            print("CSV file for domain %s has been successfully updated" % domain)





if __name__ == "__main__":
    core_nlp()
    dbd()
    refinamento()

