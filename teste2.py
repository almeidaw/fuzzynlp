from pycorenlp import StanfordCoreNLP
import pandas as pd
import os
import pathlib

# Define local do servidor StanforCoreNLPToolkit
nlp = StanfordCoreNLP('http://localhost:9000')
polarity = 0
feature = ""
# the number of documents of the training set, restricted to the domain D(i), in which feature C occurs
s = 0
# index referring to the domain D(i) which the feature belongs to
index=0
auxindex=index
auxdoc=0
# Define o caminho para os diretórios dos domínios de treinamento (inModel) e de teste (outModel)
inModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/dranziera')
# outModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/out_of_domain')

# pega o nome de cada diretório dentro do diretório Dranziera
for dir in os.listdir(inModelDir):

	# transforma esse nome do diretório em uma path para verificar se é um diretório mesmo
	# (para evitar de pegar arquivos ocultos do sistema que podem estar dentro do diretório Dranziera)
	dirPath = os.path.join(inModelDir, dir)
	if os.path.isdir(dirPath):

		# para verificar em qual dominio estamos
		index+=1
		print(index)
		auxindex=index
	
		# lista todos os arquivos dentro do diretório do domínio
		files = os.listdir(dirPath)
		
		# abre cada arquivo na pasta do domínio para leitura
		for file in files:
			
			with open("%s/%s" % (dirPath, file)) as dataset:

				# lê primeira linha do arquivo em questão e itera por cada uma das próximas linhas (cada linha é um review)
				line = dataset.readline()
				while line:

					# joga linha toda no Stanford Core NLP Toolkit
					coreoutput = nlp.annotate(line, properties={'annotators': 'lemma, pos, sentiment', 'outputFormat': 'json', 'timeout': 100000})

					print("\rDOCUMENT: %s" % (line))
                                        
					auxdoc+=1
					
					for sentence in coreoutput["sentences"]:
						
						print("SENTENCE: %d\n" % (sentence["index"]))

						for token in sentence["tokens"]:
							# senticnetoutput = sn.concept(line)
							# Extração de caracteristicas- simples e complexa
							# caso seja o primeiro doc então pega a primeira caracteristica que atende aos requisitos
							if feature == "":
								if token["pos"] == "RB" or token["pos"] == "RBR" or token["pos"] == "RBS" or token["pos"] == "NN" or token["pos"] == "NNS" or token["pos"] == "NNP" or token["pos"] == "VB" or token["pos"] == "VBD" or token["pos"] == "VBG" or token["pos"] == "VBN" or token["pos"] == "VBP" or token["pos"] == "JJS" or token["pos"] == "JJR" or token["pos"] == "JJ":
									feature = token["word"]
									# primeira ocorrência da feature no doc
									s+=1
									if sentence["sentiment"] == "Positive":
										polarity+=1	
									elif sentence["sentiment"] == "Negative":
										polarity-=1
								
							elif feature == token["word"]:
								# if s != auxdoc:
									# s+=1
								if sentence["sentiment"] == "Positive":
									polarity+=1	
								elif sentence["sentiment"] == "Negative":
									polarity-=1

							print("TOKEN: %d, WORD: %s, LEMMA: %s, POS: %s, SENTIMENT: %s" % (token["index"], token["word"], token["lemma"], token["pos"], sentence["sentiment"]))
						# with open(file +'.txt', 'w') as outfile:
    							# json.dump(coreoutput,outfile)
						
						print("\r")


					line = dataset.readline()
			if s == 0:
				p=0
			else:
				p= polarity/s
			print("Preliminary Polarity: %d, Dominio: %s" % (p, auxindex))
			with open('teste' + file +'.txt', 'w') as arquivo:
				arquivo.write("Feature: %s, Variable S: %d,  Variable Polarity: %d, Preliminary Polarity: %d, Dominio: %s" % (feature, s, polarity, p, auxindex))
			print("\r")

		# quando o algoritmo vai para a outra pasta, significa que é um novo dominio. Então, as variaveis são reinicializadas.
		polarity=0
		s=0
		auxdoc=0
			

    				


	#dt.to_csv('CoreNLPOutput.csv’)






