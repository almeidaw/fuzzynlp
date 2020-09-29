from pycorenlp import StanfordCoreNLP
import os
import pathlib

#Define local do servidor StanforCoreNLPToolkit
nlp = StanfordCoreNLP('http://localhost:9000')

#Define o caminho para os diretórios dos domínios de treinamento (inModel) e de teste (outModel)
inModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/dranziera')
outModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/out_of_domain')

#pega o nome de cada diretório dentro do diretório Dranziera
for dir in os.listdir(inModelDir):

	#print para separar os domínios em caso de testes
	#print("\n\n######################################## %s ########################################\n\n" % (dir))

	#transforma esse nome do diretório em uma path para verificar se é um diretório mesmo 
	#(para evitar de pegar arquivos ocultos do sistema que podem estar dentro do diretório Dranziera)
	dirPath = os.path.join(inModelDir, dir)
	if os.path.isdir(dirPath):

		#lista todos os arquivos dentro do diretório do domínio
		files = os.listdir(dirPath)

		#abre cara arquivo na pasta do domínio para leitura
		for file in files:

			#print para separar os arquivos em caso de testes
			#print("\n\n######################################## %s ########################################\n\n" % (file))

			with open("%s/%s" % (dirPath, file)) as dataset:

				#lê primeira linha do arquivo em questão e itera por cada uma das próximas linhas (cada linha é um review)
				line = dataset.readline()
				while line:

					#joga linha toda no Stanford Core NLP Toolkit
					coreoutput = nlp.annotate(line, properties={'annotators': 'sentiment', 'outputFormat': 'json', 'timeout': 100000})

					for s in coreoutput["sentences"]:
						print("%d: '%s': %s %s" % (s["index"], " ".join([t["word"] for t in s["tokens"]]),s["sentimentValue"], s["sentiment"]))

					line = dataset.readline()