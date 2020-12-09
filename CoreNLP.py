import json
from pycorenlp import StanfordCoreNLP
import pandas as pd
from senticnet.senticnet import SenticNet
import numpy as np
import pysentiment2 as ps
import time
import os

in_domain_dir = 'dataset/in_domain'
out_domain_dir = 'dataset/out_of_domain'

training_jsons_dir = 'outputs/training/jsons'
training_features_dir = 'outputs/training/features'

testing_jsons_dir = 'outputs/testing/jsons'
testing_features_dir = 'outputs/testing/features'


def core_nlp_json(domain_dir, jsons_dir, port):
    nlp = StanfordCoreNLP('http://localhost:'+str(port))
    # Rodar o servidor com:
    # java -mx5g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000

    # Verifica se o caminho de origem fornecido é um diretório e se não é oculto,
    # lista os arquivos do diretório e os ordena por ordem alfabética
    if os.path.isdir(domain_dir) and domain_dir[0] != ".":
        txt_files = os.listdir(domain_dir)
        txt_files.sort()

        # Itera por cada arquivo no diretório e verifica se já existe um aquivo json correspondente
        for txt_file in txt_files:
            j_file = f"{txt_file[:-3]}json"
            if j_file in os.listdir(jsons_dir):
                continue

            # Se o arquivo não for oculto ele é aberto e cada um de seus documentos (linhas) é passada para o
            # CoreNLP Toolkit. Os resultados são guardados em uma lista de dicionários
            docs = []
            if not txt_file.startswith('.'):
                print("Processando CoreNLP para documentos do arquivo %s" % txt_file)
                try:
                    with open("%s/%s" % (domain_dir, txt_file)) as dataset:
                        doc = dataset.readline()
                        while doc:
                            try:
                                core_output = nlp.annotate(doc,
                                                           properties={'annotators': 'lemma, pos, depparse',
                                                                       'outputFormat': 'json', 'timeout': 100000})

                                docs.append({"document": doc, "corenlp": core_output})
                                doc = dataset.readline()
                            except:
                                print("Não foi possível realizar chamada para CoreNLP")

                except:
                    print("Não foi possível abrir arquivo %s" % txt_file)

            # Tenta escrever o resultado em um arquivo JSON de mesmo nome do arquivo de entrada
            try:
                with open(os.path.join(jsons_dir, j_file), 'w+') as json_file:
                    json.dump(docs, json_file)
            except:
                print("Não foi possível escrever arquivo %s" % j_file)
                print(docs)
            else:
                print("Arquivo %s escrito com sucesso" % j_file)
    else:
        print("Caminho especificado não é um diretório")


def parse_json(jsons_dir, features_test_dir, domain_name, file_test):
    if True: # file_simple not in os.listdir(festures_test_dir) or file_complex not in os.listdir(festures_test_dir):

        # Listas de features buscadas
        nouns = ("NN", "NNS", "NNP", "NNPS")
        adjectives = ("JJS", "JJR", "JJ")
        verbs = ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ")
        adverbs = ("RB", "RBR", "RBS")

        lemmas_simple = []
        tokens_simple = []
        pos_simple = []
        dependencies = []
        lemmas_complex_gov = []
        lemmas_complex_dep = []
        tokens_complex_gov = []
        tokens_complex_dep = []
        pos_complex_gov = []
        pos_complex_dep = []
        kic_simple = []
        kic_complex = []
        sic_simple = []
        sic_complex = []
        zic_simple = []
        zic_complex = []
        ni_list_simple = []
        ni_list_complex = []
        difreq_simple = []
        difreq_complex = []
        peic_simple = []
        peic_complex = []

        # Verifica se o caminho de origem fornecido é um diretório e se não é oculto,
        # lista os arquivos do diretório e os ordena por ordem alfabética
        if os.path.isdir(jsons_dir) and jsons_dir[0] != ".":
            files = os.listdir(jsons_dir)
            files.sort()

            # Itera por cada arquivo no diretório e verifica se já existe um aquivo json correspondente
            for file in files:
                if str(file_test) in file:
                    continue
                if not file.startswith('.'):
                    print("Processando features para documentos do arquivo %s" % file)
                    # try:
                    with open("%s/%s" % (jsons_dir, file)) as json_file:
                        json_list = json.load(json_file)
                        for json_feature in json_list:

                            # Conjunto de features do documento
                            in_document_simple = set()
                            in_document_complex = set()
                            index_in_document_simple = set()
                            index_in_document_complex = set()

                            # Verifica se o arquivo (e portanto o documento) é positivo ou negativo
                            if "neg" in file:
                                polarity = -1
                            else:
                                polarity = 1

                            # Extração de features simples
                            for sentence in json_feature["corenlp"]["sentences"]:
                                for token in sentence["tokens"]:
                                    lemma = token["lemma"]
                                    pos = token["pos"]
                                    word = token["word"]
                                    if pos in nouns or pos in adjectives or pos in verbs or pos in adverbs:

                                        if not (lemma in lemmas_simple):
                                            lemmas_simple.append(lemma)
                                            tokens_simple.append(word)
                                            kic_simple.append(polarity)
                                            sic_simple.append(1)
                                            zic_simple.append(1)
                                            pos_simple.append(pos)
                                            in_document_simple.add(lemma)
                                            index_in_document_simple.add(len(lemmas_simple)-1)

                                        else:
                                            index = lemmas_simple.index(lemma)
                                            zic_simple[index] += 1
                                            kic_simple[index] += polarity
                                            index_in_document_simple.add(index)
                                            if lemma not in in_document_simple:
                                                sic_simple[index] += 1
                                                in_document_simple.add(lemma)

                                for dependency in sentence["basicDependencies"]:
                                    if not dependency["dep"] == "ROOT" and not "punct" == dependency["dep"]:
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
                                            lemma_governor = governor["lemma"]
                                            lemma_dependent = dependent["lemma"]
                                            token_governor = governor["word"]
                                            token_dependent = dependent["word"]
                                            dependency = dependency["dep"]

                                            # Se essa feature ja tiver sido processada
                                            if lemma_governor in lemmas_complex_gov and \
                                                    lemma_dependent in lemmas_complex_dep and \
                                                    lemmas_complex_gov.index(lemma_governor) == \
                                                    lemmas_complex_dep.index(lemma_dependent):
                                                index = lemmas_complex_gov.index(lemma_governor)
                                                zic_complex[index] += 1
                                                kic_complex[index] += polarity
                                                index_in_document_complex.add(index)
                                                if str([lemma_governor, lemma_dependent]) not in in_document_complex:
                                                    sic_complex[index] += 1
                                                    in_document_complex.add(str([lemma_governor, lemma_dependent]))

                                            else:
                                                lemmas_complex_gov.append(lemma_governor)
                                                lemmas_complex_dep.append(lemma_dependent)
                                                tokens_complex_gov.append(token_governor)
                                                tokens_complex_dep.append(token_dependent)
                                                dependencies.append(dependency)
                                                kic_complex.append(polarity)
                                                sic_complex.append(1)
                                                zic_complex.append(1)
                                                pos_complex_gov.append(pos_gov)
                                                pos_complex_dep.append(pos_dep)
                                                in_document_complex.add(str([lemma_governor, lemma_dependent]))
                                                index_in_document_complex.add(len(lemmas_complex_gov) - 1)

                            del json_feature["corenlp"]
                            json_feature.update({"feature_simple": list(in_document_simple),
                                                 "feature_complex": list(in_document_complex),
                                                 "index_simples": list(index_in_document_simple),
                                                 "index_complex": list(index_in_document_complex)})

                        try:
                            with open(os.path.join(features_test_dir, "features.%s" % file), 'w+') as json_features:
                                json.dump(json_list, json_features)
                        except:
                            print("Não foi possível escrever arquivo features.%s" % file)
                            print(json_list)
                        else:
                            print("Arquivo features.%s escrito com sucesso" % file)

            ni = sum(zic_simple)+sum(zic_complex)
            for index_lemma in range(len(lemmas_simple)):
                peic = kic_simple[index_lemma] / sic_simple[index_lemma]
                peic_simple.append(peic)
                difreq_simple.append(zic_simple[index_lemma] / ni)
                ni_list_simple.append(ni)

            for index_lemma in range(len(lemmas_complex_gov)):
                peic = kic_complex[index_lemma] / sic_complex[index_lemma]
                peic_complex.append(peic)
                difreq_complex.append(zic_complex[index_lemma] / ni)
                ni_list_complex.append(ni)

        domain_simple = pd.DataFrame(
            list(zip(tokens_simple, lemmas_simple, pos_simple, kic_simple, sic_simple, peic_simple, zic_simple,
                     ni_list_simple, difreq_simple)),
            columns=["TOKEN", "FEATURE", "POS", "POLARIDADES (K)", "DOCS COM FEATURE (S)", "POLARIDADE ESTIMADA (P=K/S)",
                     "FEATURE NO DOMINIO (Z)", "TODAS FEATURES NO DOMINIO (N)", "RELEVANCIA NO DOMINIO (FREQ=Z/N)"])

        domain_complex = pd.DataFrame(
            list(zip(tokens_complex_gov, tokens_complex_dep, lemmas_complex_gov, lemmas_complex_dep, dependencies,
                     pos_complex_gov, pos_complex_dep, kic_complex, sic_complex, peic_complex, zic_complex,
                     ni_list_complex, difreq_complex)),
            columns=["TOKEN GOV", "TOKEN DEP", "FEATURE GOV", "FEATURE DEP", "DEPENDÊNCIA", "POS GOV", "POS DEP",
                     "POLARIDADES (K)", "DOCS COM FEATURE (S)", "POLARIDADE ESTIMADA (P=K/S)", "FEATURE NO DOMINIO (Z)",
                     "TODAS FEATURES NO DOMINIO (N)", "RELEVANCIA NO DOMINIO (FREQ=Z/N)"])

        try:
            domain_simple.to_csv("%s/simple.%s.csv" % (features_test_dir, domain_name))
        except:
            print("Não foi possível gravar arquivo CSV com features simples do domínio %s" % domain_name)
            print(domain_simple)
        else:
            print("Arquivo CSV com features simples do domínio %s gravado com sucesso" % domain_name)

        try:
            domain_complex.to_csv("%s/complex.%s.csv" % (features_test_dir, domain_name))
        except:
            print("Não foi possível gravar arquivo CSV com features complexas do domínio %s" % domain_name)
            print(domain_complex)
        else:
            print("Arquivo CSV com features complexas do domínio %s gravado com sucesso" % domain_name)

    else:
        print("Domínio já processado")


def dbd(features_fold_dir):
    if os.path.isdir(features_fold_dir) and features_fold_dir[0] != ".":
        domains = os.listdir(features_fold_dir)
        domains.sort()

        features_simples = {}
        features_complex = {}
        start = time.time()
        for domain in domains:
            features_fold_domain_dir = os.path.join(os.path.join(features_fold_dir, domain))
            if features_fold_domain_dir and domain[0] != ".":
                print("\nExtraindo features de", domain)
                try:
                    file_simple = pd.read_csv(f"{features_fold_domain_dir}/simple.{domain}.csv", index_col=0)
                    file_complex = pd.read_csv(f"{features_fold_domain_dir}/complex.{domain}.csv", index_col=0)
                except:
                    continue

                print("\nsimples\n")
                for index, row in file_simple.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features simples", end='\r')
                    zic = row["FEATURE NO DOMINIO (Z)"]
                    feature = row["FEATURE"]
                    if feature in features_simples.keys():
                        features_simples[feature] = features_simples.get(feature) + zic
                    else:
                        features_simples[feature] = zic

                print("\n\ncomplexas\n")
                for index, row in file_complex.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features complexas", end='\r')
                    zic = row["FEATURE NO DOMINIO (Z)"]
                    feature_gov = row["FEATURE GOV"]
                    feature_dep = row["FEATURE DEP"]
                    feature = str(feature_gov) + str(feature_dep)
                    if feature in features_complex.keys():
                        features_complex[feature] = features_complex.get(feature) + zic
                    else:
                        features_complex[feature] = zic
        end = time.time()
        print("\nTempo decorrido na extração: %d minutos" % ((end - start) / 60))

        start = time.time()
        for domain in domains:
            features_fold_domain_dir = os.path.join(os.path.join(features_fold_dir, domain))
            if features_fold_domain_dir and domain[0] != ".":
                print("\nAgregando DBD do dominio", domain)
                try:
                    file_simple = pd.read_csv(f"{features_fold_domain_dir}/simple.{domain}.csv", index_col=0)
                    file_complex = pd.read_csv(f"{features_fold_domain_dir}/complex.{domain}.csv", index_col=0)
                except:
                    continue

                zjc_simple = []
                zjc_complex = []
                diuniq_simple = []
                diuniq_complex = []
                dbd_simple = []
                dbd_complex = []

                print("\nsimples\n")
                for index, row in file_simple.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features simples", end='\r')
                    zic = row["FEATURE NO DOMINIO (Z)"]
                    feature = row["FEATURE"]
                    difreq = row["RELEVANCIA NO DOMINIO (FREQ=Z/N)"]

                    zjc = features_simples.get(feature)
                    diuniq = zic/zjc
                    dbd = diuniq*difreq

                    zjc_simple.append(zjc)
                    diuniq_simple.append(diuniq)
                    dbd_simple.append(dbd)

                print("\n\ncomplexas\n")
                for index, row in file_complex.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features complexas", end='\r')
                    zic = row["FEATURE NO DOMINIO (Z)"]
                    feature_gov = row["FEATURE GOV"]
                    feature_dep = row["FEATURE DEP"]
                    feature = str(feature_gov) + str(feature_dep)
                    difreq = row["RELEVANCIA NO DOMINIO (FREQ=Z/N)"]

                    zjc = features_complex.get(feature)
                    diuniq = zic / zjc
                    dbd = diuniq * difreq

                    zjc_complex.append(zjc)
                    diuniq_complex.append(diuniq)
                    dbd_complex.append(dbd)

                file_simple['FEATURE NOS DOMINIOS (Zjc)'] = zjc_simple
                file_complex['FEATURE NOS DOMINIOS (Zjc)'] = zjc_complex
                file_simple['UNIQUENESS (UNIQ=Z/Zjc)'] = diuniq_simple
                file_complex['UNIQUENESS (UNIQ=Z/Zjc)'] = diuniq_complex
                file_simple['DBD (DBD=UNIQ*FREQ)'] = dbd_simple
                file_complex['DBD (DBD=UNIQ*FREQ)'] = dbd_complex

                try:
                    file_simple.to_csv("%s/simple.%s.csv" % (features_fold_domain_dir, domain))
                except:
                    print(f"\nNão foi possível gravar arquivo CSV com features simples do domínio {domain}")
                    print(file_simple)
                else:
                    print(f"\nArquivo CSV com features simples do domínio {domain} gravado com sucesso")

                try:
                    file_complex.to_csv(f"{features_fold_domain_dir}/complex.{domain}.csv")
                except:
                    print(f"Não foi possível gravar arquivo CSV com features complexas do domínio {domain}")
                    print(file_complex)
                else:
                    print(f"Arquivo CSV com features complexas do domínio {domain} gravado com sucesso")
        end = time.time()
        print("Tempo decorrido na agregação: %d minutos" % ((end - start) / 60))


def refinement_sentic(s_path):
    if os.path.isdir(s_path) and s_path[0] != ".":
        domains = os.listdir(s_path)
        domains.sort()

        sn = SenticNet()
        features_simples = {}
        features_complex = {}

        start = time.time()
        for domain in domains:
            dir_domain = os.path.join(os.path.join(s_path, domain))
            if dir_domain and domain[0] != ".":
                print(f"\nExtraindo senticnet para features do domínio {domain}")
                try:
                    file_simple = pd.read_csv("%s/simple.%s.csv" % (dir_domain, domain), index_col=0)
                    file_complex = pd.read_csv("%s/complex.%s.csv" % (dir_domain, domain), index_col=0)
                except:
                    continue

                sentics_simple = []
                sentics_complex = []

                print("\nsimples\n")
                for index, row in file_simple.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features simples", end='\r')

                    feature = row["FEATURE"]

                    if feature in features_simples.keys():
                        sentic = features_simples.get(feature)
                    else:
                        try:
                            sentic = float(sn.polarity_intense(feature))
                            if sentic > 1:
                                print(feature, ":", sentic)
                        except:
                            sentic = 0

                        features_simples[feature] = sentic

                    sentics_simple.append(sentic)

                print("\n\ncomplexas\n")
                for index, row in file_complex.iterrows():
                    if index % 10000 == 0:
                        print(f"{index} features complexas", end='\r')

                    feature_gov = row["FEATURE GOV"]
                    feature_dep = row["FEATURE DEP"]
                    feature = str(feature_gov) + str(feature_dep)

                    if feature in features_complex.keys():
                        sentic = features_complex.get(feature)
                    else:
                        try:
                            sentic_gov = float(sn.polarity_intense(feature_gov))
                            if sentic_gov > 1:
                                print(feature, ":", sentic_gov)
                        except:
                            sentic_gov = 0
                        try:
                            sentic_dep = float(sn.polarity_intense(feature_dep))
                            if sentic_dep > 1:
                                print(feature, ":", sentic_dep)
                        except:
                            sentic_dep = 0

                        dependencia = row["DEPENDÊNCIA"]

                        if sentic_gov != 0:
                            if sentic_dep != 0:
                                if dependencia == "advmod":
                                    sentic = sentic_gov * sentic_dep
                                else:
                                    sentic = np.mean([sentic_gov, sentic_dep])
                            else:
                                sentic = sentic_gov
                        else:
                            if sentic_dep != 0:
                                sentic = sentic_dep
                            else:
                                sentic = 0

                        features_complex[feature] = sentic

                    sentics_complex.append(sentic)

                file_simple['SENTICNET'] = sentics_simple
                file_complex['SENTICNET'] = sentics_complex

                try:
                    file_simple.to_csv(f"{dir_domain}/simple.{domain}.csv")
                except:
                    print(f"\nNão foi possível gravar arquivo CSV com features simples do domínio {domain}")
                    print(file_simple)
                else:
                    print(f"\nArquivo CSV com features simples do domínio %s gravado com sucesso {domain}")

                try:
                    file_complex.to_csv(f"{dir_domain}/complex.{domain}.csv")
                except:
                    print(f"Não foi possível gravar arquivo CSV com features complexas do domínio {domain}")
                    print(file_complex)
                else:
                    print(f"Arquivo CSV com features complexas do domínio {domain} gravado com sucesso")

        end = time.time()
        print(f"Tempo decorrido na agregação: {(end - start) / 60} minutos")


def refinement_general(s_path):
    if os.path.isdir(s_path) and s_path[0] != ".":
        domains = os.listdir(s_path)
        domains.sort()

        features_simples = {}
        features_complex = {}

        start = time.time()
        for domain in domains:
            dir_domain = os.path.join(os.path.join(s_path, domain))
            if dir_domain and domain[0] != ".":
                print(f"\nExtraindo general inquirer para features do domínio {domain}")
                try:
                    file_simple = pd.read_csv(f"{dir_domain}/simple.{domain}.csv", index_col=0)
                    file_complex = pd.read_csv(f"{dir_domain}/complex.{domain}.csv", index_col=0)
                except:
                    continue

                generals_simple = []
                generals_complex = []

                print("\nsimples\n")
                for index, row in file_simple.iterrows():
                    if index % 100 == 0:
                        print(f"{index} features simples", end='\r')

                    feature = row["FEATURE"]

                    if feature in features_simples.keys():
                        general = features_simples.get(feature)
                    else:
                        try:
                            general = float(ps.HIV4().get_score(ps.HIV4().tokenize(feature)).get("Polarity"))
                        except:
                            general = 0

                        features_simples[feature] = general

                    generals_simple.append(general)

                print("\n\ncomplexas\n")
                for index, row in file_complex.iterrows():
                    if index % 100 == 0:
                        print(f"{index} features complexas", end='\r')

                    feature_gov = row["FEATURE GOV"]
                    feature_dep = row["FEATURE DEP"]
                    feature = str(feature_gov) + str(feature_dep)

                    if feature in features_complex.keys():
                        general = features_complex.get(feature)
                    else:
                        try:
                            general_gov = float(ps.HIV4().get_score(ps.HIV4().tokenize(feature)).get("Polarity"))
                        except:
                            general_gov = 0
                        try:
                            general_dep = float(ps.HIV4().get_score(ps.HIV4().tokenize(feature)).get("Polarity"))
                        except:
                            general_dep = 0

                        dependencia = row["DEPENDÊNCIA"]

                        if general_gov != 0:
                            if general_dep != 0:
                                if dependencia == "advmod":
                                    general = general_gov * general_dep
                                else:
                                    general = np.mean([general_gov, general_dep])
                            else:
                                general = general_gov
                        else:
                            if general_dep != 0:
                                general = general_dep
                            else:
                                general = 0

                        features_complex[feature] = general

                    generals_complex.append(general)

                file_simple['GENERAL INQUIRER'] = generals_simple
                file_complex['GENERAL INQUIRER'] = generals_complex

                try:
                    file_simple.to_csv(f"{dir_domain}/simple.{domain}.csv")
                except:
                    print(f"\nNão foi possível gravar arquivo CSV com features simples do domínio {domain}")
                    print(file_simple)
                else:
                    print(f"\nArquivo CSV com features simples do domínio {domain} gravado com sucesso")

                try:
                    file_complex.to_csv(f"{dir_domain}/complex.{domain}.csv")
                except:
                    print(f"Não foi possível gravar arquivo CSV com features complexas do domínio {domain}")
                    print(file_complex)
                else:
                    print(f"Arquivo CSV com features complexas do domínio {domain} gravado com sucesso")

        end = time.time()
        print(f"Tempo decorrido na agregação: {(end - start) / 60} minutos")


def trapezoid(features_fold_dir):
    if os.path.isdir(features_fold_dir) and features_fold_dir[0] != ".":
        domains = os.listdir(features_fold_dir)
        domains.sort()
        start = time.time()
        for domain in domains:
            features_fold_domain_dir = os.path.join(os.path.join(features_fold_dir, domain))
            if features_fold_domain_dir and domain[0] != ".":
                print(f"Calculando trapezoide das features do domínio {domain}")

                averages_simple = []
                averages_complex = []
                variances_simple = []
                variances_complex = []
                as_simple = []
                bs_simple = []
                cs_simple = []
                ds_simple = []
                as_complex = []
                bs_complex = []
                cs_complex = []
                ds_complex = []

                try:
                    file_simple = pd.read_csv(f"{features_fold_domain_dir}/simple.{domain}.csv", index_col=0)
                except:
                    print("Não foi possível abrir arquivo de features simples")
                else:
                    print("\nFeatures simples")
                    for index, row in file_simple.iterrows():
                        if index % 10000 == 0:
                            print(f"{index} features simples", end='\r')
                        peic = row["POLARIDADE ESTIMADA (P=K/S)"]
                        sentic = row["SENTICNET"]
                        general = row["GENERAL INQUIRER"]
                        if sentic != 0 and -1 <= sentic <= 1:
                            if general != 0 and -1 <= general <= 1:
                                average = np.mean([peic, sentic, general])
                                variance = np.var([peic, sentic, general])
                            else:
                                average = np.mean([peic, sentic])
                                variance = np.var([peic, sentic])
                        else:
                            if general != 0 and -1 <= general <= 1:
                                average = np.mean([peic, general])
                                variance = np.var([peic, general])
                            else:
                                average = np.mean([peic])
                                variance = np.var([peic])

                        b = min([peic, average])
                        a = max([-1, b - variance])
                        c = max([peic, average])
                        d = min([1, c + variance])

                        averages_simple.append(average)
                        variances_simple.append(variance)
                        bs_simple.append(b)
                        as_simple.append(a)
                        cs_simple.append(c)
                        ds_simple.append(d)

                try:
                    file_complex = pd.read_csv(f"{features_fold_domain_dir}/complex.{domain}.csv", index_col=0)
                except:
                    print("Não foi possível abrir arquivo de features complexas")
                else:
                    print("\ncomplexas\n")
                    for index, row in file_complex.iterrows():
                        if index % 10000 == 0:
                            print(f"{index} features complexas", end='\r')
                        peic = row["POLARIDADE ESTIMADA (P=K/S)"]
                        sentic = row["SENTICNET"]
                        general = row["GENERAL INQUIRER"]

                        if sentic != 0 and -1 <= sentic <= 1:
                            if general != 0 and -1 <= general <= 1:
                                average = np.mean([peic, sentic, general])
                                variance = np.var([peic, sentic, general])
                            else:
                                average = np.mean([peic, sentic])
                                variance = np.var([peic, sentic])
                        else:
                            if general != 0 and -1 <= general <= 1:
                                average = np.mean([peic, general])
                                variance = np.var([peic, general])
                            else:
                                average = np.mean([peic])
                                variance = np.var([peic])

                        b = min([peic, average])
                        a = max([-1, b - variance])
                        c = max([peic, average])
                        d = min([1, c + variance])

                        averages_complex.append(average)
                        variances_complex.append(variance)
                        bs_complex.append(b)
                        as_complex.append(a)
                        cs_complex.append(c)
                        ds_complex.append(d)

                file_simple['MÉDIA DAS POLARIDADES'] = averages_simple
                file_complex['MÉDIA DAS POLARIDADES'] = averages_complex
                file_simple['VARIÂNCIA DA POLARIDADES'] = variances_simple
                file_complex['VARIÂNCIA DA POLARIDADES'] = variances_complex
                file_simple['A'] = as_simple
                file_complex['A'] = as_complex
                file_simple['B'] = bs_simple
                file_complex['B'] = bs_complex
                file_simple['C'] = cs_simple
                file_complex['C'] = cs_complex
                file_simple['D'] = ds_simple
                file_complex['D'] = ds_complex

                try:
                    file_simple.to_csv("%s/simple.%s.csv" % (features_fold_domain_dir, domain))
                except:
                    print("\nNão foi possível gravar arquivo CSV com features simples do domínio %s" % domain)
                    print(file_simple)
                else:
                    print("\nArquivo CSV com features simples do domínio %s gravado com sucesso" % domain)

                try:
                    file_complex.to_csv("%s/complex.%s.csv" % (features_fold_domain_dir, domain))
                except:
                    print("Não foi possível gravar arquivo CSV com features complexas do domínio %s" % domain)
                    print(file_complex)
                else:
                    print("Arquivo CSV com features complexas do domínio %s gravado com sucesso" % domain)
        end = time.time()
        print("Tempo decorrido na agregação: %d minutos" % ((end - start) / 60))


def aggregation(training_jsons_dir, training_features_fold_dir, testing_jsons_dir, testing_features_fold_dir, fold):
    start = time.time()
    if os.path.isdir(training_features_fold_dir) and training_features_fold_dir[0] != ".":
        training_domains = os.listdir(training_features_fold_dir)
        training_domains.sort()
        domains = {}
        for training_domain in training_domains:
            features = {}
            features_fold_domain_dir = os.path.join(os.path.join(training_features_fold_dir, training_domain))
            if features_fold_domain_dir and training_domain[0] != ".":
                print(f"Processando features do domínio {training_domain}")
                try:
                    file_simple = pd.read_csv(f"{features_fold_domain_dir}/simple.{training_domain}.csv", index_col=0)
                except:
                    print("Não foi possível abrir arquivo de features simples")
                else:
                    print("\nFeatures simples")
                    for index, row in file_simple.iterrows():
                        if index % 10000 == 0:
                            print(f"{index} features simples", end='\r')
                        feature = row["FEATURE"]
                        a = row["A"]
                        b = row["B"]
                        c = row["C"]
                        d = row["D"]
                        dbd = row["DBD (DBD=UNIQ*FREQ)"]
                        features[feature] = {"a": a, "b": b, "c": c, "d": d, "dbd": dbd}

                try:
                    file_complex = pd.read_csv(f"{features_fold_domain_dir}/complex.{training_domain}.csv", index_col=0)
                except:
                    print("Não foi possível abrir arquivo de features complexas")
                else:
                    print("\ncomplexas\n")
                    for index, row in file_complex.iterrows():
                        if index % 10000 == 0:
                            print(f"{index} features complexas", end='\r')
                        feature_gov = row["FEATURE GOV"]
                        feature_dep = row["FEATURE DEP"]
                        feature = str(feature_gov)+str(feature_dep)
                        a = row["A"]
                        b = row["B"]
                        c = row["C"]
                        d = row["D"]
                        dbd = row["DBD (DBD=UNIQ*FREQ)"]
                        features[feature] = {"a": a, "b": b, "c": c, "d": d, "dbd": dbd}

            domains[training_domain] = features

    if os.path.isdir(testing_jsons_dir) and testing_jsons_dir[0] != ".":
        testing_domains = os.listdir(testing_jsons_dir)
        testing_domains.sort()

        for testing_domain in testing_domains:
            testing_domain_dir = os.path.join(testing_jsons_dir, testing_domain)
            if os.path.isdir(testing_domain_dir) and testing_domain[0] != ".":
                testing_files = os.listdir(testing_domain_dir)
                testing_files.sort()

                # Itera por cada arquivo no diretório e verifica se já existe um aquivo json correspondente
                for testing_file in testing_files:
                    if not testing_file.startswith('.'):
                        print(f"Processando features para documentos do arquivo {testing_file}")
                        with open(f"{testing_domain_dir}/{testing_file}") as testing_json_file:
                            testing_json_list = json.load(testing_json_file)
                            documents = []
                            for index in range(len(testing_json_list)):
                                testing_json_document = testing_json_list[index]
                                if index % 100 == 0:
                                    print(f"{index} documentos", end='\r')

                                dbds = []
                                polarities = []

                                for domain in domains.keys():
                                    a_list = []
                                    b_list = []
                                    c_list = []
                                    d_list = []
                                    dbd_list = []
                                    for sentence in testing_json_document["corenlp"]["sentences"]:
                                        for token in sentence["tokens"]:
                                            feature = token["lemma"]
                                            if feature in domains.get(domain).keys():
                                                values = domains.get(domain).get(feature)
                                                a_list.append(values.get("a"))
                                                b_list.append(values.get("b"))
                                                c_list.append(values.get("c"))
                                                d_list.append(values.get("d"))
                                                dbd_list.append(values.get("dbd"))

                                        for dependency in sentence["basicDependencies"]:
                                            if not dependency["dep"] == "ROOT" and not "punct" == dependency["dep"]:
                                                dependent = sentence["tokens"][dependency["dependent"] - 1]["lemma"]
                                                governor = sentence["tokens"][dependency["governor"] - 1]["lemma"]
                                                feature = str(governor)+str(dependent)
                                                if feature in domains.get(domain).keys():
                                                    values = domains.get(domain).get(feature)
                                                    a_list.append(values.get("a"))
                                                    b_list.append(values.get("b"))
                                                    c_list.append(values.get("c"))
                                                    d_list.append(values.get("d"))
                                                    dbd_list.append(values.get("dbd"))

                                    a = np.mean(a_list)
                                    b = np.mean(b_list)
                                    c = np.mean(c_list)
                                    d = np.mean(d_list)
                                    dbd = np.mean(dbd_list)

                                    polarity = np.mean([float(c), float(d)])

                                    dbds.append(dbd)
                                    polarities.append(polarity)

                                semifinal = [dbds[i] * polarities[i] for i in range(len(dbds))]
                                final = sum(semifinal)
                                documents.append(final)

                            final_df = pd.DataFrame(documents, columns=["VALORES"])

                            testing_features_fold_domain_dir = os.path.join(testing_features_fold_dir, testing_domain)

                            try:
                                final_df.to_csv(f"{testing_features_fold_domain_dir}/{testing_file[:-4]}csv")
                            except:
                                print(f"Não foi possível gravar arquivo CSV com polaridades do domínio {testing_domain}")
                                print(documents)
                            else:
                                print(f"Arquivo CSV com polaridades do domínio {testing_domain} gravado com sucesso")

    if os.path.isdir(training_jsons_dir) and training_jsons_dir[0] != ".":
        testing_domains = os.listdir(training_jsons_dir)
        testing_domains.sort()

        for testing_domain in testing_domains:
            testing_domain_dir = os.path.join(training_jsons_dir, testing_domain)
            if os.path.isdir(testing_domain_dir) and testing_domain[0] != ".":
                testing_files = os.listdir(testing_domain_dir)
                testing_files.sort()

                # Itera por cada arquivo no diretório e verifica se já existe um aquivo json correspondente
                for testing_file in testing_files:
                    if not testing_file.startswith('.') and str(fold) in testing_file:
                        print(f"Processando features para documentos do arquivo {testing_file}")
                        with open(f"{testing_domain_dir}/{testing_file}") as testing_json_file:
                            testing_json_list = json.load(testing_json_file)
                            documents = []
                            for index in range(len(testing_json_list)):
                                testing_json_document = testing_json_list[index]
                                if index % 100 == 0:
                                    print(f"{index} documentos", end='\r')

                                dbds = []
                                polarities = []

                                for domain in domains.keys():
                                    a_list = []
                                    b_list = []
                                    c_list = []
                                    d_list = []
                                    dbd_list = []
                                    for sentence in testing_json_document["corenlp"]["sentences"]:
                                        for token in sentence["tokens"]:
                                            feature = token["lemma"]
                                            if feature in domains.get(domain).keys():
                                                values = domains.get(domain).get(feature)
                                                a_list.append(values.get("a"))
                                                b_list.append(values.get("b"))
                                                c_list.append(values.get("c"))
                                                d_list.append(values.get("d"))
                                                dbd_list.append(values.get("dbd"))

                                        for dependency in sentence["basicDependencies"]:
                                            if not dependency["dep"] == "ROOT" and not "punct" == dependency["dep"]:
                                                dependent = sentence["tokens"][dependency["dependent"] - 1]["lemma"]
                                                governor = sentence["tokens"][dependency["governor"] - 1]["lemma"]
                                                feature = str(governor)+str(dependent)
                                                if feature in domains.get(domain).keys():
                                                    values = domains.get(domain).get(feature)
                                                    a_list.append(values.get("a"))
                                                    b_list.append(values.get("b"))
                                                    c_list.append(values.get("c"))
                                                    d_list.append(values.get("d"))
                                                    dbd_list.append(values.get("dbd"))

                                    a = np.mean(a_list)
                                    b = np.mean(b_list)
                                    c = np.mean(c_list)
                                    d = np.mean(d_list)
                                    dbd = np.mean(dbd_list)

                                    polarity = (float(a)*0 + float(b)*1 + float(c)*1 + float(d)*1)/(0+1+1+0)

                                    dbds.append(dbd)
                                    polarities.append(polarity)

                                semifinal = [dbds[i] * polarities[i] for i in range(len(dbds))]
                                final = sum(semifinal)
                                documents.append(final)

                            final_df = pd.DataFrame(documents, columns=["VALORES"])

                            try:
                                final_df.to_csv(f"{training_features_fold_dir}/{testing_file[:-4]}csv")
                            except:
                                print(f"Não foi possível gravar arquivo CSV com polaridades do domínio {testing_domain}")
                                print(documents)
                            else:
                                print(f"Arquivo CSV com polaridades do domínio {testing_domain} gravado com sucesso")

    end = time.time()
    print("Tempo decorrido na agregação: %d minutos" % ((end - start) / 60))


def resultados(testing_features_dir):
    for fold in range(5):
        testing_features_fold_dir = os.path.join(testing_features_dir, f"Fold {fold}")
        resultados = {}
        if os.path.isdir(testing_features_fold_dir):
            for testing_domain in os.listdir(testing_features_fold_dir):
                testing_features_fold_domain_dir = os.path.join(testing_features_fold_dir, testing_domain)
                if os.path.isdir(testing_features_fold_domain_dir) and testing_domain[0] != ".":
                    for testing_file in os.listdir(testing_features_fold_domain_dir):
                        if not testing_file.startswith('.'):
                            domain_df = resultados.get(testing_file)
                            if domain_df is not None:
                                domain_df.add(pd.read_csv(f"{testing_features_fold_domain_dir}/{testing_file}", index_col=0))
                            else:
                                resultados[testing_file] = pd.read_csv(f"{testing_features_fold_domain_dir}/{testing_file}", index_col=0)

    for df in resultados.keys():
        resultado_final = resultados.get(df)
        try:
            resultado_final.to_csv(f"{testing_features_dir}/{df}.csv")
        except:
            print(f"Não foi possível gravar arquivo CSV com polaridades do domínio {df}")
            print(resultado_final)
        else:
            print(f"Arquivo CSV com polaridades do domínio {df} gravado com sucesso")


if __name__ == "__main__":
    # CoreNLP
    features_fold_list = os.listdir(in_domain_dir)
    features_fold_list.sort()

    for domain in features_fold_list:
        in_domain_domain_dir = os.path.join(in_domain_dir, domain)

        if os.path.isdir(in_domain_domain_dir) and domain[0] != ".":

            if not os.path.exists(os.path.join(training_jsons_dir, domain)):
                os.makedirs(os.path.join(training_jsons_dir, domain))

            jsons_domain_dir = os.path.join(training_jsons_dir, domain)

            start = time.time()
            core_nlp_json(in_domain_domain_dir, jsons_domain_dir, 9000)
            end = time.time()

            print(f"Tempo decorrido: {(end-start)/60} minutos")

    # Extração de features
    for fold in range(5):
        print(f"Processando arquivo da fold {fold}")

        if not os.path.exists(os.path.join(training_features_dir, f"Fold {fold}")):
            os.makedirs(os.path.join(training_features_dir, f"Fold {fold}"))

        features_fold_dir = os.path.join(training_features_dir, f"Fold {fold}")

        features_fold_list = os.listdir(features_fold_dir)
        features_fold_list.sort()

        for domain in features_fold_list:

            json_domain_dir = os.path.join(training_jsons_dir, domain)

            if not os.path.exists(os.path.join(features_fold_dir, domain)):
                os.makedirs(os.path.join(features_fold_dir, domain))

            features_fold_domain_dir = os.path.join(features_fold_dir, domain)

            start = time.time()
            parse_json(json_domain_dir, features_fold_domain_dir, domain, fold)
            end = time.time()
            print(f"Tempo decorrido: {(end - start) / 60} minutos")

    # Passa apenas a pasta do fold
    for fold in range(5):
        print(f"Processando arquivo da fold {fold}")
        if not os.path.exists(os.path.join(training_features_dir, f"Fold {fold}")):
            os.makedirs(os.path.join(training_features_dir, f"Fold {fold}"))
        training_features_fold_dir = os.path.join(training_features_dir, f"Fold {fold}")

        if not os.path.exists(os.path.join(testing_features_dir, f"Fold {fold}")):
            os.makedirs(os.path.join(testing_features_dir, f"Fold {fold}"))
        testing_features_fold_dir = os.path.join(testing_features_dir, f"Fold {fold}")

        start = time.time()
        dbd(training_features_fold_dir)
        refinement_sentic(training_features_fold_dir)
        refinement_general(training_features_fold_dir)
        trapezoid(training_features_fold_dir)
        aggregation(training_jsons_dir, training_features_fold_dir, testing_jsons_dir, testing_features_fold_dir, fold)
        end = time.time()
        print(f"Tempo decorrido: {(end - start) / 60} minutos")

    resultados(testing_features_dir)