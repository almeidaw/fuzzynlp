from pycorenlp import StanfordCoreNLP
#from senticnet.senticnet import SenticNet
import os
import pathlib
import pandas as pd
import json

#Define local do servidor StanforCoreNLPToolkit
nlp = StanfordCoreNLP('http://localhost:9000')
#sn = SenticNet()

#Define o caminho para os diretórios dos domínios de treinamento (inModel) e de teste (outModel)
inModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/dranziera')
outModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/out_of_domain')

#pega o nome de cada diretório dentro do diretório Dranziera
for dir in os.listdir(inModelDir):

	#transforma esse nome do diretório em uma path para verificar se é um diretório mesmo 
	#(para evitar de pegar arquivos ocultos do sistema que podem estar dentro do diretório Dranziera)
	dirPath = os.path.join(inModelDir, dir)
	if os.path.isdir(dirPath):

		#lista todos os arquivos dentro do diretório do domínio
		files = os.listdir(dirPath)

		#abre cara arquivo na pasta do domínio para leitura
		for file in files:

			with open("%s/%s" % (dirPath, file)) as dataset:

				#lê primeira linha do arquivo em questão e itera por cada uma das próximas linhas (cada linha é um review)
				line = dataset.readline()
				while line:

					#joga linha toda no Stanford Core NLP Toolkit
					coreoutput = nlp.annotate(line, properties={'annotators': 'lemma, pos, sentiment', 'outputFormat': 'json', 'timeout': 100000})

					print("\rDOCUMENT: %s" % (line))

					for sentence in coreoutput["sentences"]:
						
						print("SENTENCE: %d\n" % (sentence["index"]))

						for token in sentence["tokens"]:
							#senticnetoutput = sn.concept(line)
							print("TOKEN: %d, WORD: %s, LEMMA: %s, POS: %s, SENTIMENT: %s" % (token["index"], token["word"], token["lemma"], token["pos"], sentence["sentiment"]))
						with open(file +'.txt', 'w') as outfile:
    							json.dump(coreoutput,outfile)

						with open(file +'.txt', 'r') as infile:
    							for line in infile:
        							if "C:\\Users\\larys\\OneDrive\\Área de Trabalho\\Trabalho Prático- Inteligência Computacional" in line:
            								next(infile)
            								extractData = [next(infile).strip().replace('"', "") for i in range(3)]
            								for i in extractData:
               									print("{}={}".format(*i.split(" : ")))

						
						print("\r")
						

					line = dataset.readline()


	#dt.to_csv('CoreNLPOutput.csv’)






