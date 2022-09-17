#-*- coding: utf-8-*-
import sys
import codecs 
import nltk
import math
from nltk import bigrams
from nltk import trigrams

#funzione che restituisce la lista totale di token e la lista in cui ad ogni token è associata la sua POS
def AnnotazioneLinguistica(frasi):
	tokensTOT = []
	tokensPOStot = []
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		#pos_tag prende in input una lista di token ed esegue l'analisi morfo-sintattica
		tokensPOS = nltk.pos_tag(tokens)
		tokensTOT = tokensTOT + tokens
		tokensPOStot = tokensPOStot + tokensPOS
	return tokensTOT, tokensPOStot

#funzione che restituisce la lista delle POS
def EstraiPOS(TestoAnalizzatoPOS):
	ListaPOS = []
	for bigramma in TestoAnalizzatoPOS:
		#inserisco il secondo elemento del bigramma (ossia la POS) all'interno della lista POS
		ListaPOS.append(bigramma[1])
	return ListaPOS

#funzione che restituisce la lista dei token senza punteggiatura
def EstraiListaParoleSenzaPunteggiatura (tokens):
	ListaParole=[]
	for parola in tokens:
		#se il token non è tra quelli inseriti tra [], allora aggiungo il token alla lista 
		if (parola not in [",",".",";",":","(",")","!","?","-"]):
			ListaParole.append(parola)
	return ListaParole

#funzione che restituisce la lista con i bigrammi che non contengono punteggiatura, articoli e congiunzioni
def EstraiBigrammi(TestoAnalizzatoPOS):
	BigrammiCorretti=[]
	#estraggo bigrammi di token in cui ad ogni token corrisponde la sua POS
	bigrammi = list(bigrams(TestoAnalizzatoPOS))
	for bigramma in bigrammi:
		#verifico che la POS di ogni token che compone il bigramma non sia tra le congiunzioni, gli articoli e la punteggiatura
		if ((bigramma[0][1] not in ["CC", "IN", "DT", ".",";",":",",","(",")","!","?"]) and (bigramma[1][1] not in ["CC", "IN", "DT", ".",";",":",",","(",")","!","?"])):
			#se la condizione è verificata inserisco i token che compongono il bigramma all'interno della lista BigrammiCorretti
			BigrammiCorretti.append((bigramma[0][0],bigramma[1][0]))
	return BigrammiCorretti

#funzione che restituisce la lista con i trigrammi che non contengono punteggiatura, articoli e congiunzioni
def EstraiTrigrammi(TestoAnalizzatoPOS):
	TrigrammiCorretti=[]
	#estraggo trigrammi di token in cui ad ogni token corrisponde la sua POS
	trigrammi= trigrams(TestoAnalizzatoPOS)
	for trigramma in trigrammi:
		#verifico che la POS di ogni token che compone il trigramma non sia tra le congiunzioni, gli articoli e la punteggiatura
		if ((trigramma[0][1] not in ["CC", "IN", "DT", ".",";",":",",","(",")","!","?"]) and (trigramma[1][1] not in ["CC", "IN", "DT", ".",";",":",",","(",")","!","?"]) and (trigramma[2][1] not in ["CC", "IN", "DT", ".",";",":",",","(",")","!","?"])):
			#se la condizione è verificata inserisco i token che compongono il trigramma all'interno della lista TrigrammiCorretti
			TrigrammiCorretti.append((trigramma[0][0],trigramma[1][0],trigramma[2][0]))
	return TrigrammiCorretti

#funzione che restituisce la lista con i bigrammi formati da Aggettivo-Sostantivo o Sostantivo-Aggettivo
def EstraiBigrammiAggSost(TestoAnalizzatoPOS, TestoTokenizzato):
	BigrammiAggSost = []
	bigrammiAnalizzati = list(bigrams(TestoAnalizzatoPOS))
	distribuzione = nltk.FreqDist(TestoTokenizzato)
	for bigramma in bigrammiAnalizzati:
		#verifico che le parole che compongono il bigramma abbiano frequenza maggiore di 2
		if ((distribuzione[bigramma[0][0]] > 2) and (distribuzione[bigramma[1][0]] > 2)):
			#verifico che il bigramma sia composto da un sostantivo e un aggettivo
			if (((bigramma[0][1] in ["NN", "NNS", "NNP", "NNPS"]) and (bigramma[1][1] in ["JJ","JJS","JJR"])) or ((bigramma[0][1] in ["JJ","JJS","JJR"]) and (bigramma[1][1] in  ["NN","NNS", "NNP","NNPS"]))):
				BigrammiAggSost.append((bigramma[0][0],bigramma[1][0])) #inserisco il bigramma nella lista che contiene i bigrammi composti da aggettivo e sostantivo
	return BigrammiAggSost

#funzione che restituisce un dizionario in cui ad ogni bigramma (aggettivo-sostantivo) è associata la sua probabilità congiunta
def CreaDizionarioBigProbCongiunta(BigAggSost,TestoTokenizzato):
	Dizionario = {}
	bigrammiTipo = set(BigAggSost)
	for big in bigrammiTipo:
		probCongiunta= probCong(big,TestoTokenizzato) #per calcolare la probabilità congiunta utilizzo la funzione probCong
		Dizionario[big] = probCongiunta #associo ad ogni bigramma la sua probabilità congiunta
	return Dizionario

#funzione che calcola la probabilità congiunta 
def probCong(big,TestoTokenizzato):
	#individuo i bigrammi del testo
	bigrammiTesto = list(bigrams(TestoTokenizzato))
	#calcolo la frequenza dei bigrammi del testo
	distribuzione = nltk.FreqDist(bigrammiTesto)
	#calcolo la frequenza del bigramma preso in considerazione
	freqBig=distribuzione[big]
	lunghezza = len(TestoTokenizzato)
	probCongiunta= (freqBig*1.0)/(lunghezza*1.0) #calcolo la probabilità dividendo la frequenza del bigramma per la lunghezza del corpus
	return probCongiunta

#funzione che restituisce un dizionario in cui ad ogni bigramma (aggettivo-sostantivo) è associata la sua probabilità condizionata
def CreaDizionarioBigProbCondizionata(BigAggSost,TestoTokenizzato): 
	Dizionario = {}
	bigrammiTipo = set(BigAggSost)
	for big in bigrammiTipo:
		probCondizionata = probCond(big,TestoTokenizzato) #per calcolare la probabilità condizionata utilizzo la funzione probCond
		Dizionario[big] = probCondizionata #associo ad ogni bigramma la sua probabilità condizionata
	return Dizionario

#funzione che calcola la probabilità condizionata
def probCond(big, TestoTokenizzato):
	#individuo i bigrammi del testo
	bigrammiTesto = list(bigrams(TestoTokenizzato))
	#calcolo la frequenza dei bigrammi del testo
	distribuzione = nltk.FreqDist(bigrammiTesto)
	#calcolo la frequenza del bigramma preso in considerazione
	freqBig=distribuzione[big]
	#calcolo la frequenza della prima parola del bigramma
	freqA= TestoTokenizzato.count(big[0])  #big[0] è il primo token del bigramma
	probCondizionata= (freqBig*1.0)/(freqA*1.0) #calcolo la probabilità dividendo la frequenza del bigramma per la frequenza del primo elemento del bigramma
	return probCondizionata

#funzione che restituisce un dizionario in cui ad ogni bigramma (aggettivo-sostantivo) è associata la sua LMI
def CreaDizionarioBigLMI(BigAggSost,TestoTokenizzato):
	Dizionario = {}
	bigrammiTipo= set(BigAggSost)
	for big in bigrammiTipo:
		LMI= CalcolaLMI(big,TestoTokenizzato) #per calcolare la probabilità condizionata utilizzo la funzione CalcolaLMI
		Dizionario[big] = LMI #associo ad ogni bigramma la sua LMI
	return Dizionario

#funzione che calcola la LMI
def CalcolaLMI(big,TestoTokenizzato):
	#individuo i bigrammi del testo
	bigrammiTesto = list(bigrams(TestoTokenizzato))
	#calcolo la loro distribuzione
	distribuzione = nltk.FreqDist(bigrammiTesto)
	#calcolo la frequenza del bigramma preso in considerazione
	freqBig=distribuzione[big]
	probCongiunta= probCong(big,TestoTokenizzato) #calcolo la probabilità congiunta del bigramma, sfruttando la funzione probCong
	freqA= TestoTokenizzato.count(big[0]) #frequenza del primo token
	freqB= TestoTokenizzato.count(big[1]) #frequenza del secondo token
	probA= (freqA*1.0)/(len(TestoTokenizzato)*1.0) #probabilità primo token 
	probB= (freqB*1.0)/(len(TestoTokenizzato)*1.0) #probabilità secondo token 
	probCongiuntaInd= (probA)*(probB) #calcolo la probabilità congiunta di eventi indipendenti
	MI= math.log(((probCongiunta*1.0)/(probCongiuntaInd*1.0)),2)
	#calcolo la LMI
	LMI= freqBig*MI
	return LMI

#funzione che restituisce la lista di frasi richieste 
def TrovaFrasi(frasi, TestoTokenizzato):
	#Le frasi devono essere lunghe almeno 10 token e ogni token deve avere una frequenza maggiore di 2
	ListaFrasi = []
	distribuzione = nltk.FreqDist(TestoTokenizzato)
	for frase in frasi:
		tokens = nltk.word_tokenize(frase)
		conta = 0 #variabile che utilizzo per contare i token > 2
		lunghezza = len(tokens)
		#individuo le frasi che sono lunghe almeno 10 token
		if (lunghezza >= 10):
			for tok in tokens:
				#verifico che la frequenza del token sia maggiore di 2
				if distribuzione[tok]>2:
					#per ogni token che ha frequenza maggiore di 2, aumento di uno la variabile conta
					conta = conta + 1
			if conta == lunghezza: #se conta è uguale alla lunghezza della frase, allora tutti i token che compongono la frase hanno frequenza maggiore di 2
				ListaFrasi.append(frase) #dunque aggiungo la frase che rispetta le condizioni sopra verificate alla ListaFrasi
	return ListaFrasi

#funzione che restituisce la frase con probabilità massima calcolata attraverso un modello di Markov di ordine 0 e la sua probabilità
def TrovaFraseMarkov0(FrasiCorrette, TestoTokenizzato):
	probMax = 0.0 
	fraseMax = ""
	lunghezzaCorpus = len(TestoTokenizzato)
	#calcolo la distribuzione dei token, che sfrutterò per il calcolo della probabilità del token
	distribuzione = nltk.FreqDist(TestoTokenizzato)
	for frase in FrasiCorrette:
		probFrase = 1.0 #variabile che utilizzerò per calcolarmi la probabilità della frase
		tokens = nltk.word_tokenize(frase)
		for tok in tokens:
			probabilitaTok = (distribuzione[tok]*1.0)/(lunghezzaCorpus*1.0) #calcolo la probabilità del singolo token
			probFrase = probabilitaTok * probFrase #calcolo la probabilità della frase
		if probFrase > probMax: #se la probabilità della frase è maggiore della probabilità massima, allora la probabilità massima diventa la probabilità della frase e la frase con probabilità massima diventa quella frase
			probMax = probFrase
			fraseMax = frase
	return fraseMax, probMax

#funzione che restituisce la frase con probabilità massima calcolata attraverso un modello di Markov di ordine 1 e la sua probabilità
def TrovaFraseMarkov1(FrasiCorrette, TestoTokenizzato):
	probMax = 0.0
	fraseMax = ""
	lunghezzaCorpus = len(TestoTokenizzato)
	bigrammi = list(bigrams(TestoTokenizzato))
	distribuzioneBigrammi = nltk.FreqDist(bigrammi) #frequenza bigrammi nel testo
	distribuzioneToken = nltk.FreqDist(TestoTokenizzato) #frequenza dei token nel testo
	for frase in FrasiCorrette:
		probFrase = 1.0
		tokens = nltk.word_tokenize(frase)
		PrimoToken = tokens[0]
		prob1 = (distribuzioneToken[PrimoToken]*1.0)/(lunghezzaCorpus*1.0) #calcolo la probabilità del primo token
		bigrammiFrase = list(bigrams(tokens)) #individuo i bigrammi che compongono la frase
		for big in bigrammiFrase: 
			probabilita = (distribuzioneBigrammi[big]*1.0)/(distribuzioneToken[big[0]]*1.0) 
			probFrase = probabilita * probFrase 
		probFraseTOT = probFrase * prob1 #calcolo la probabilità della frase
		if probFraseTOT > probMax: #se la probabilità della frase è maggiore della probabilità massima, allora la probabilità massima diventa la probabilità della frase e la frase con probabilità massima diventa quella frase
			probMax = probFraseTOT
			fraseMax = frase
	return fraseMax, probMax

#funzione che utilizzo per ordinare i dizionari
def Ordina(dict):
	return sorted(dict.items(), key=lambda x: x[1], reverse= True)

#funzione che restituisce la lista dei nomi propri di persona, la lista dei nomi di luogo e la lista delle entità
def AnalisiEntita(TestoAnalizzatoPOS):
	ListaEntita= []
	ListaNNP=[] #lista nomi propri di persona
	ListaNL = [] #lista nomi luogo
	analisi1 = nltk.ne_chunk(TestoAnalizzatoPOS) 
	for nodo in analisi1: #ciclo l'albero scorrendo i nodi
		NNP=""
		NL=""
		NE=""
		if hasattr(nodo, 'label'): 
			if nodo.label() in ["PERSON"]: #estraggo l'etichetta dal nodo
				for partNNP in nodo.leaves(): #ciclo le foglie del nodo selezionato
					NNP=NNP+" "+partNNP[0]
				ListaNNP.append(NNP) #inserisco i nomi propri di persona all'interno della lista ListaNNP
			if nodo.label() in ["GPE"]:
				for partNL in nodo.leaves(): #ciclo le foglie del nodo selezionato
					NL=NL+" "+partNL[0]
				ListaNL.append(NL) #inserisco i nomi di luogo all'interno della lista ListaNL
			if nodo.label() in ["PERSON","GPE","ORGANIZATION"]:
				for partNE in nodo.leaves(): #ciclo le foglie del nodo selezionato
					NE=NE+" "+partNE[0]
				ListaEntita.append(NE) #inserisco tutte le entità nominate all'interno della lista ListaEntita
	return ListaNNP, ListaNL, ListaEntita

def main(file1,file2):
	fileInput1 = codecs.open(file1, "r", "utf-8")
	raw1 = fileInput1.read()
	fileInput2 = codecs.open(file2, "r", "utf-8")
	raw2 = fileInput2.read()
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	#divido i testi in frasi
	frasi1= sent_tokenizer.tokenize(raw1)
	frasi2= sent_tokenizer.tokenize(raw2)
	#tokenizzo e compio il pos tagging dei due testi mediante la funzione AnnotazioneLinguistica
	TestoTokenizzato1, TestoAnalizzatoPOS1 = AnnotazioneLinguistica(frasi1)
	TestoTokenizzato2, TestoAnalizzatoPOS2 = AnnotazioneLinguistica(frasi2)
	#estraggo la lista delle POS 
	ListaPOS1 = EstraiPOS(TestoAnalizzatoPOS1)
	ListaPOS2 = EstraiPOS(TestoAnalizzatoPOS2)
	
	print"1) estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa frequenza"
	print
	#le 10 POS più frequenti
	print "a) le 10 POS più frequenti nel corpus", file1, ":"
	#calcolo la frequenza di ogni POS
	FreqPOS1= nltk.FreqDist(ListaPOS1)
	#individuo le 10 POS più frequenti
	Prime10POS1= FreqPOS1.most_common(10)
	for POS1 in Prime10POS1:
		print " POS:", POS1[0].encode("utf-8"),"--->", "frequenza:", POS1[1]
	print
	#faccio lo stesso per il secondo file
	print "a) le 10 POS più frequenti nel corpus", file2, ":"
	FreqPOS2= nltk.FreqDist(ListaPOS2)
	Prime10POS2= FreqPOS2.most_common(10)
	for POS2 in Prime10POS2:
		print " POS:", POS2[0].encode("utf-8"),"--->", "frequenza:", POS2[1]
	print
	print
	#estraete i primi 20 token più frequenti ad esclusione della punteggiatura
	#estraggo i token senza punteggiatura
	ListaParole1= EstraiListaParoleSenzaPunteggiatura(TestoTokenizzato1)
	ListaParole2= EstraiListaParoleSenzaPunteggiatura(TestoTokenizzato2)

	print "b) i primi 20 token più frequenti - ad esclusione della punteggiatura - nel corpus", file1, ":"
	#calcolo la frequenza di ogni token
	FreqListaParole1= nltk.FreqDist(ListaParole1)
	#individuo i 20 token più frequenti
	Prime20Parole1= FreqListaParole1.most_common(20)
	for Parole1 in Prime20Parole1:
		print " token:", Parole1[0].encode("utf-8"),"--->", "frequenza:", Parole1[1]
	print
	#faccio lo stesso per il secondo file
	print "b) i primi 20 token più frequenti - ad esclusione della punteggiatura - nel corpus", file2, ":"
	FreqListaParole2= nltk.FreqDist(ListaParole2)
	Prime20Parole2= FreqListaParole2.most_common(20)
	for Parole2 in Prime20Parole2:
		print " token:", Parole2[0].encode("utf-8"),"--->", "frequenza:", Parole2[1]
	print
	print
	#estraggo i bigrammi che non contengono  punteggiatura, articoli e congiunzioni
	Bigrammi1 = EstraiBigrammi(TestoAnalizzatoPOS1)
	Bigrammi2 = EstraiBigrammi(TestoAnalizzatoPOS2)
	print "c) i primi 20 bigrammi di token più frequenti - che non contengono punteggiatura, articoli e congiunzioni- nel corpus", file1, ":"
	#calcolo la frequenza di ogni bigramma
	FreqBigrammi1= nltk.FreqDist(Bigrammi1)
	#individuo i 20 bigrammi più frequenti
	Primi20Bigrammi1= FreqBigrammi1.most_common(20)
	for Bigrammi1 in Primi20Bigrammi1:
		print " bigramma:", Bigrammi1[0][0].encode("utf-8"),Bigrammi1[0][1].encode("utf-8"),"--->", "frequenza:", Bigrammi1[1]
	print
	#faccio lo stesso per il secondo file
	print "c) i primi 20 bigrammi di token più frequenti - che non contengono punteggiatura, articoli e congiunzioni- nel corpus", file2, ":"
	FreqBigrammi2= nltk.FreqDist(Bigrammi2)
	Primi20Bigrammi2= FreqBigrammi2.most_common(20)
	for Bigrammi2 in Primi20Bigrammi2:
		print " bigramma:", Bigrammi2[0][0].encode("utf-8"),Bigrammi2[0][1].encode("utf-8"),"--->", "frequenza:", Bigrammi2[1]
	print
	print
	#estraggo i trigrammi che non contengono  punteggiatura, articoli e congiunzioni
	Trigrammi1 = EstraiTrigrammi(TestoAnalizzatoPOS1)
	Trigrammi2 = EstraiTrigrammi(TestoAnalizzatoPOS2)
	print "d) i primi 20 trigrammi di token più frequenti - che non contengono punteggiatura, articoli e congiunzioni - nel corpus", file1, ":"
	#calcolo la frequenza di ogni trigramma
	FreqTrigrammi1= nltk.FreqDist(Trigrammi1)
	#individuo i 20 trigrammi più frequenti
	Primi20FreqTrigrammi1= FreqTrigrammi1.most_common(20)
	for Trigrammi1 in Primi20FreqTrigrammi1:
		print " trigramma:", Trigrammi1[0][0].encode("utf-8"),Trigrammi1[0][1].encode("utf-8"),Trigrammi1[0][2].encode("utf-8"),"--->", "frequenza:", Trigrammi1[1]
	print
	#faccio lo stesso per il secondo file
	print "d) i primi 20 trigrammi di token più frequenti - che non contengono punteggiatura, articoli e congiunzioni - nel corpus", file2, ":"
	FreqTrigrammi2= nltk.FreqDist(Trigrammi2)
	Primi20FreqTrigrammi2= FreqTrigrammi2.most_common(20)
	for Trigrammi2 in Primi20FreqTrigrammi2:
		print " trigramma:", Trigrammi2[0][0].encode("utf-8"),Trigrammi2[0][1].encode("utf-8"),Trigrammi2[0][2].encode("utf-8"),"--->", "frequenza:", Trigrammi2[1]
	print
	print "---------------------------------------------"
	print
	print "2) estraete ed ordinate i 20 bigrammi composti da Aggettivo e Sostantivo (dove ogni token deve avere una frequenza maggiore di 2):"
	print
	#estraggo i bigrammi composti da Aggettivo-Sostantivo o Sostantivo-Aggettivo
	BigAggSost1= EstraiBigrammiAggSost(TestoAnalizzatoPOS1, TestoTokenizzato1)	
	BigAggSost2= EstraiBigrammiAggSost(TestoAnalizzatoPOS2, TestoTokenizzato2)
	#creo un dizionario in cui ad ogni bigramma Aggettivo-Sostantivo o Sostantivo-Aggettivo corrisponde una probabilità congiunta
	DizionarioBigProbCongiunta1 = CreaDizionarioBigProbCongiunta(BigAggSost1,TestoTokenizzato1) 
	DizionarioBigProbCongiunta2 = CreaDizionarioBigProbCongiunta(BigAggSost2,TestoTokenizzato2)
	#ordino il dizionario
	OrdinatoDizionarioBigProbCongiunta1 = Ordina(DizionarioBigProbCongiunta1)
	OrdinatoDizionarioBigProbCongiunta2 = Ordina(DizionarioBigProbCongiunta2)
	print "a) con probabilità congiunta massima nel corpus", file1, ":"
	#stampo i primi 20 elementi del dizionario per ottenere i primi 20 bigrammi con probabilità congiunta massima
	for i in OrdinatoDizionarioBigProbCongiunta1[0:20]:
		print " bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","probabilità congiunta:",i[1]
	print
	print "a) con probabilità congiunta massima nel corpus", file2, ":"
	for i in OrdinatoDizionarioBigProbCongiunta2[0:20]:
		print " bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","probabilità congiunta:",i[1]
	print 
	print
	#creo un dizionario in cui ad ogni bigramma Aggettivo-Sostantivo o Sostantivo-Aggettivo corrisponde una probabilità condizionata
	DizionarioBigProbCondizionata1 = CreaDizionarioBigProbCondizionata(BigAggSost1,TestoTokenizzato1) 
	DizionarioBigProbCondizionata2 = CreaDizionarioBigProbCondizionata(BigAggSost2,TestoTokenizzato2)
	#lo ordino
	OrdinatoDizionarioBigProbCondizionata1 = Ordina(DizionarioBigProbCondizionata1)
	OrdinatoDizionarioBigProbCondizionata2 = Ordina(DizionarioBigProbCondizionata2)
	print "b) con probabilità condizionata massima nel corpus", file1, ":"
	#stampo i primi 20 elementi del dizionario per ottenere i primi 20 bigrammi con probabilità condizionata massima
	for i in OrdinatoDizionarioBigProbCondizionata1[0:20]:
		print " bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","probabilità condizionata:",i[1]
	print
	print "b) con probabilità condizionata massima nel corpus", file2, ":"
	for i in OrdinatoDizionarioBigProbCondizionata2[0:20]:
		print " bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","probabilità condizionata:",i[1]
	print
	print
	#creo un dizionario in cui ad ogni bigramma Aggettivo-Sostantivo o Sostantivo-Aggettivo corrisponde la forza associativa calcolata con la LMI
	DizionarioBigLMI1 = CreaDizionarioBigLMI(BigAggSost1,TestoTokenizzato1)
	DizionarioBigLMI2 = CreaDizionarioBigLMI(BigAggSost2, TestoTokenizzato2)
	#lo ordino
	OrdinatoDizionarioBigLMI1= Ordina(DizionarioBigLMI1)
	OrdinatoDizionarioBigLMI2= Ordina(DizionarioBigLMI2)
	print "c) con LMI massima nel corpus", file1, ":"
	#stampo i primi 20 elementi del dizionario per ottenere i primi 20 bigrammi con LMI massima
	for i in OrdinatoDizionarioBigLMI1[0:20]:
		print "bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","LMI:",i[1]
	print
	print "c) con LMI massima nel corpus", file2, ":"
	for i in OrdinatoDizionarioBigLMI2[0:20]:
		print "bigramma:", i[0][0].encode("utf-8"), i[0][1].encode("utf-8"),"--->","LMI:",i[1]
	print
	print "---------------------------------------------"
	print
	print "3) le due frasi con probabilità più alta:"
	print
	#le due frasi con probabilità più alta. Dove la probabilità della prima frase deve essere calcolata attraverso un modello di Markov di ordine 0 mentre la seconda con un modello di Markov di ordine 1
	#individuo le frasi lunghe 10 token e ogni token con frequenza maggiore di 2
	FrasiCorrette1 = TrovaFrasi(frasi1, TestoTokenizzato1)
	FrasiCorrette2 = TrovaFrasi(frasi2, TestoTokenizzato2)
	#individuo la frase con probabilità massima calcolata con un modello di Markov di ordine 0 e la sua probabilità
	Frase1MaxMarkov0, ProbMax1Markov0 = TrovaFraseMarkov0(FrasiCorrette1, TestoTokenizzato1)
	Frase2MaxMarkov0, ProbMax2Markov0  = TrovaFraseMarkov0(FrasiCorrette2, TestoTokenizzato2)
	print "a) nel corpus", file1, "la frase con probabilità più alta calcolata con un modello di Markov di ordine 0 è:"
	print Frase1MaxMarkov0.encode("utf-8"),"--->", "con probabilità:", ProbMax1Markov0
	print
	print "a) nel corpus", file2, "la frase con probabilità più alta calcolata con un modello di Markov di ordine 0 è:"
	print Frase2MaxMarkov0.encode("utf-8"), "--->","con probabilità:", ProbMax2Markov0
	print
	print
	#individuo la frase con probabilità massima calcolata con un modello di Markov di ordine 1 e la sua probabilità
	Frase1MaxMarkov1, ProbMax1Markov1 = TrovaFraseMarkov1(FrasiCorrette1, TestoTokenizzato1)
	Frase2MaxMarkov1, ProbMax2Markov1  = TrovaFraseMarkov1(FrasiCorrette2, TestoTokenizzato2)
	print "b) nel corpus", file1, "la frase con probabilità più alta calcolata con un modello di Markov di ordine 1 è:"
	print Frase1MaxMarkov1.encode("utf-8"),"--->", "con probabilità:", ProbMax1Markov1
	print
	print "b) nel corpus", file2, "la frase con probabilità più alta calcolata con un modello di Markov di ordine 1 è:"
	print Frase2MaxMarkov1.encode("utf-8"),"--->", "con probabilità:", ProbMax2Markov1
	print
	print "---------------------------------------------"
	print
	#ENTITA NOMINATE
	print "4) dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:"
	print
	#Estraggo la lista dei nomi propri di persona, la lista dei nomi di luogo e la lista delle entità
	ListaNNP1, ListaNL1, ListaEntita1 = AnalisiEntita(TestoAnalizzatoPOS1)
	ListaNNP2, ListaNL2, ListaEntita2 = AnalisiEntita(TestoAnalizzatoPOS2)
	print "a) i 20 nomi propri di persona più frequenti (tipi), ordinati per frequenza nel corpus", file1,":"
	#calcolo la frequenza di ogni nome proprio di persona
	FreqNomiPropriPersona1= nltk.FreqDist(ListaNNP1)
	#individuo i primi 20 nomi propri di persona più frequenti
	PrimiVentiNomiPropri1= FreqNomiPropriPersona1.most_common(20)
	#li stampo
	for nomiP1 in PrimiVentiNomiPropri1:
		print " nome proprio di persona:", nomiP1[0].encode("utf-8"),"--->", "frequenza:", nomiP1[1]
	print
	print "b) i 20 nomi propri di luogo più frequenti (tipi), ordinati per frequenza nel corpus", file1, ":"
	#calcolo la frequenza di ogni nome di luogo
	FreqNomiLuogo1= nltk.FreqDist(ListaNL1)
	#individuo i primi 20 nomi propri di luogo più frequenti
	PrimiVentiNomiLuogo1= FreqNomiLuogo1.most_common(20)
	#li stampo
	for nomiL1 in PrimiVentiNomiLuogo1:
		print "nome luogo:", nomiL1[0].encode("utf-8"),"--->", "frequenza:", nomiL1[1]
	print 
	print
	print "a) i 20 nomi propri di persona più frequenti (tipi), ordinati per frequenza nel corpus", file2,":"
	#faccio lo stesso per il secondo file
	#calcolo la frequenza di ogni nome proprio di persona
	FreqNomiPropriPersona2= nltk.FreqDist(ListaNNP2)
	#individuo i primi 20 nomi propri di persona più frequenti
	PrimiVentiNomiPropri2= FreqNomiPropriPersona2.most_common(20)
	#li stampo
	for nomiP2 in PrimiVentiNomiPropri2:
		print "nome proprio di persona:", nomiP2[0].encode("utf-8"),"--->", "frequenza:", nomiP2[1]
	print
	print "b) i 20 nomi propri di luogo più frequenti (tipi), ordinati per frequenza nel corpus", file2, ":"
	#calcolo la frequenza di ogni nome di luogo
	FreqNomiLuogo2= nltk.FreqDist(ListaNL2)
	#individuo i primi 20 nomi propri di luogo più frequenti
	PrimiVentiNomiLuogo2= FreqNomiLuogo2.most_common(20)
	#li stampo
	for nomiL2 in PrimiVentiNomiLuogo2:
		print "nome luogo:", nomiL2[0].encode("utf-8"),"--->", "frequenza:", nomiL2[1]
		
	




main(sys.argv[1], sys.argv[2])