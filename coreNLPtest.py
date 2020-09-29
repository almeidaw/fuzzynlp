from pycorenlp import StanfordCoreNLP
import os
import pathlib

#Define local do servidor StanforCoreNLPToolkit
nlp = StanfordCoreNLP('http://localhost:9000')
#res = nlp.annotate("texto", properties={'annotators': 'depparse', 'outputFormat': 'json', 'timeout': 100000})
"""
res = nlp.annotate("The new laptop I bought is amazing. The monitor is very tidy and the new solid state drive works very well.", 
	properties={'annotators': 'depparse',
				'outputFormat': 'json',
				'timeout': 100000,
                })

"""
#for s in res["sentences"]:
#    print("%d: '%s': %s %s" % (
#        s["index"],
#        " ".join([t["word"] for t in s["tokens"]]),
#        s["sentimentValue"], s["sentiment"]))

#print(res)
"""
for s in res["sentences"]:
	for w in s["tokens"]:
		print("%s/%s " % (
		w["word"],
		w["pos"]), end='')
	print("\r")
print("\r")

for s in res["sentences"]:
	for w in s["enhancedDependencies"]:
		print("%s(%s-%d, %s-%d)" % (
		w["dep"],
		w["governorGloss"],
		w["governor"],
		w["dependentGloss"],
		w["dependent"]))
	print("\r")
print("\r")
"""

inModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/dranziera')
outModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/out_of_domain')

#print(os.listdir(dataset))   
#print(os.listdir(inModelDir),'\n\n',os.listdir(outModelDir))
#print([name for name in os.listdir(inModelDir) if os.path.isdir(os.path.join(inModelDir, name))])

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
					coreoutput = nlp.annotate(line, properties={'annotators': 'sentiment',
																'outputFormat': 'json',
																'timeout': 100000
																})

					for s in coreoutput["sentences"]:
						print("%d: '%s': %s %s" % (s["index"], " ".join([t["word"] for t in s["tokens"]]),s["sentimentValue"], s["sentiment"]))

					line = dataset.readline()
			


#complexf[]
#simplef[]

#if ()

#for (dominios de 1 a n):
#	for (features 1 a x dentro do dominio)
#		pe = soma aritmetica das polaridades da feature / n de reviews no dominio em que feature ocorre 
