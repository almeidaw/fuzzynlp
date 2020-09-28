from pycorenlp import StanfordCoreNLP
import os
import pathlib

nlp = StanfordCoreNLP('http://localhost:9000')
res = nlp.annotate("The new laptop I bought is amazing. The monitor is very tidy and the new solid state drive works very well.", 
	properties={'annotators': 'depparse',
				'outputFormat': 'json',
				'timeout': 100000,
                })
#for s in res["sentences"]:
#    print("%d: '%s': %s %s" % (
#        s["index"],
#        " ".join([t["word"] for t in s["tokens"]]),
#        s["sentimentValue"], s["sentiment"]))

#print(res)

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


inModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/dranziera')
outModelDir = pathlib.Path(__file__).parent.absolute().joinpath('paper-package/out_of_domain')

#print(os.listdir(dataset))   
#print(os.listdir(inModelDir),'\n\n',os.listdir(outModelDir))
#print([name for name in os.listdir(inModelDir) if os.path.isdir(os.path.join(inModelDir, name))])

#pega o nome de cada diretório dentro do diretório Dranziera
for dir in os.listdir(inModelDir):
	print("\n\n######################################## %s ########################################\n\n" % (dir))
	#transforma esse nome em uma path para verificar se é um diretório ou arquivo 
	#(para evitar de pegar arquivos ocultos do sistema que podem estar dentro do diretório Dranziera)
	dirPath = os.path.join(inModelDir, dir)
	if os.path.isdir(dirPath):
		files = os.listdir(dirPath)
		for file in files:
			print("\n\n######################################## %s ########################################\n\n" % (file))
			with open("%s/%s" % (dirPath, file)) as dataset:
				line = dataset.readline()
				cnt = 1
				while line:
					print("Line {}: {}".format(cnt, line.strip()))
					line = dataset.readline()
					cnt += 1
			


#complexf[]
#simplef[]

#if ()

#for (dominios de 1 a n):
#	for (features 1 a x dentro do dominio)
#		pe = soma aritmetica das polaridades da feature / n de reviews no dominio em que feature ocorre 
