# -*- coding: utf-8 -*-
import sys
import codecs
import math
import nltk

#funzione che restituisce la lunghezza del corpus, il numero delle frasi, la lista dei token e la lista in cui ad ogni token è associata la sua POS
def Analizzatore(frasi):
	lunghezzaTOT = 0 
	numeroFrasi = 0
	tokensTOT = []
	tokensPOStot = []
	for frase in frasi:
		#divido la frase in token
		tokens = nltk.word_tokenize(frase)
		#calcolo la lunghezza 
		lunghezzaTOT = lunghezzaTOT + len(tokens)
		#calcolo il numero delle frasi
		numeroFrasi = numeroFrasi + 1
		#inserisco i token in una lista
		tokensTOT = tokensTOT + tokens
		#Assegno ad ogni token la sua POS
		tokensPOS = nltk.pos_tag(tokens)
		tokensPOStot = tokensPOStot + tokensPOS
	return lunghezzaTOT, numeroFrasi, tokensTOT, tokensPOStot

#funzione che restituisce la lista di tutte le POS presenti nel corpus
def EstraiSequenzaPOS(TestoAnalizzatoPOS):
	listaPOS = []
	for bigramma in TestoAnalizzatoPOS:
		#aggiungo alla listaPOS solo la POS associata al token
		listaPOS.append(bigramma[1])
	return listaPOS

#funzione che restituisce il numero dei nomi, dei verbi, degli aggettivi, degli avverbi, delle virgole e dei punti
def ContaPOS(SequenzaPOS):
	distribuzionePOS = nltk.FreqDist(SequenzaPOS)
	numeroNomi = distribuzionePOS["NN"] + distribuzionePOS["NNS"]+distribuzionePOS["NNP"]+ distribuzionePOS["NNPS"]
	numeroVerbi = distribuzionePOS["VB"] + distribuzionePOS["VBD"] + distribuzionePOS["VBG"] + distribuzionePOS["VBN"]+distribuzionePOS["VBP"] +distribuzionePOS["VBZ"] 
	numeroAggettivi = distribuzionePOS["JJ"] + distribuzionePOS["JJR"]+distribuzionePOS["JJS"]
	numeroAvverbi = distribuzionePOS["RB"] + distribuzionePOS["RBR"]+distribuzionePOS["RBS"]
	numeroVirgole = distribuzionePOS[","]
	numeroPunti = distribuzionePOS["."]
	return numeroNomi, numeroVerbi, numeroAggettivi, numeroAvverbi, numeroVirgole, numeroPunti

#funzione che calcola per grandezze incrementali del corpus, la grandezza del vocabolario e la Type Token Ratio
def calcolaGrandezza(tokens, lunghezza):
	#calcolo le unità di migliaia della lunghezza del corpus 
	migliaia= math.floor(lunghezza/1000)
	a = 1 #a è una variabile che utilizzo per individuare le prime 1000 parole del corpus, le prime 2000 parole del corpus ecc. fino alle n migliaia della lunghezza del corpus
	while a<=migliaia:
		primeAparole = a*1000
		#individuo le parole tipo nei primi A token
		paroleTipo = set(tokens[0:primeAparole])
		#calcolo la grandezza del vocabolario
		grandezzaVoc = len(paroleTipo)
		print "nei primi", primeAparole, "token troviamo", grandezzaVoc, "parole tipo"
		#calcolo la Type Token Ratio
		ttr= (grandezzaVoc*1.0)/(primeAparole*1.0)
		print "la TTR nei primi", primeAparole, "token è:", ttr,"\n"
		a = a+1
	#se la lunghezza del corpus è diversa dalle sue n migliaia (ad esempio se 5154 è diverso da 5000), allora stampo la grandezza del vocabolario e la TTR per l'intera lunghezza del corpus
	if (lunghezza-(migliaia*1000))!= 0:
		print "in", lunghezza, "token (lunghezza totale) troviamo", len(set(tokens)), "parole tipo"
		print "in", lunghezza, "token (lunghezza totale) la TTR è:", (len(set(tokens))*1.0)/(lunghezza*1.0),"\n"

def main(file1, file2):
	fileInput1 = codecs.open(file1, "r", "utf-8")
	fileInput2= codecs.open(file2, "r", "utf-8")
	raw1 = fileInput1.read()
	raw2 = fileInput2.read()
	sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	
	#divido i due file in frasi:
	frasi1= sent_tokenizer.tokenize(raw1)
	frasi2= sent_tokenizer.tokenize(raw2)
	
	#analizzo il testo 
	lunghezza1, numeroFrasi1, tokens1, TestoAnalizzatoPOS1 = Analizzatore(frasi1)
	lunghezza2, numeroFrasi2, tokens2, TestoAnalizzatoPOS2 = Analizzatore(frasi2)
	
	#ottengo la lista di tutte le POS presenti nel corpus
	SequenzaPOS1= EstraiSequenzaPOS(TestoAnalizzatoPOS1)
	SequenzaPOS2= EstraiSequenzaPOS(TestoAnalizzatoPOS2)
	
	#estraggo i seguenti valori, utilizzando la funzione ContaPOS
	numeroNomi1, numeroVerbi1, numeroAggettivi1, numeroAvverbi1, numeroVirgole1, numeroPunti1 = ContaPOS(SequenzaPOS1)
	numeroNomi2, numeroVerbi2, numeroAggettivi2, numeroAvverbi2, numeroVirgole2, numeroPunti2 = ContaPOS(SequenzaPOS2)

	print "Confrontate i due testi sulla base delle seguenti informazioni statistiche:"
	print "1) Il numero di token"
	print
	print "il corpus", file1, "è lungo", lunghezza1, "token", "\n"
	print "il corpus", file2, "è lungo", lunghezza2, "token", "\n"
	
	#confronto i due testi
	if lunghezza1> lunghezza2:
		print "il corpus", file1, "è più lungo del corpus", file2, "\n"
	elif lunghezza1<lunghezza2:
		print "il corpus", file2, "è più lungo del corpus", file1, "\n"
	else:
		print "i due corpora hanno la stessa lunghezza"
	print
	print "------------------------------"
	print
	print "2) La lunghezza media delle frasi in termini di token"
	print
	
	#calcolo la lunghezza media delle frasi in termini di token
	LungMedia1= (lunghezza1*1.0)/(numeroFrasi1*1.0)
	LungMedia2= (lunghezza2*1.0)/(numeroFrasi2*1.0)
	print "la lunghezza media delle frasi in termini di token nel corpus", file1, "è", LungMedia1, "\n"
	print
	print "la lunghezza media delle frasi in termini di token nel corpus", file2, "è", 	LungMedia2, "\n"
	
	#confronto i due testi
	if LungMedia1 > LungMedia2:
		print "il corpus", file1, "ha una lunghezza media delle frasi in termini di token maggiore del corpus", file2, "\n"
	elif LungMedia1 < LungMedia2:
		print "il corpus", file2, "ha una lunghezza media delle frasi in termini di token maggiore del corpus", file1, "\n"
	else:
		print "i due corpora hanno la stessa lunghezza media delle frasi in termini di token"
	print
	print "------------------------------"
	print
	print "3 e 4) La grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token Ratio (TTR) all'aumento del corpus per porzioni incrementali di 1000 token"
	print
	print "nel corpus", file1, ":"
	
	#calcolo la grandezza del vocabolario e la ricchezza lessicale all'aumento del corpus
	calcolaGrandezza(tokens1,lunghezza1)
	print
	print "             -             "
	print
	print "nel corpus", file2, ":"
	calcolaGrandezza(tokens2,lunghezza2)
	print
	print "------------------------------"
	print
	print "5) Il rapporto tra sostantivi e verbi (indice che caratterizza variazioni di registro linguistico)"
	print
	
	#calcolo il rapporto tra sostantivi e verbi
	RapportoSV1= (numeroNomi1*1.0)/(numeroVerbi1*1.0)
	RapportoSV2= (numeroNomi2*1.0)/(numeroVerbi2*1.0)
	print "nel corpus", file1, "è:", RapportoSV1
	print
	print "nel corpus", file2, "è:", RapportoSV2
	print
	
	#confronto i due testi
	if RapportoSV1 > RapportoSV2:
		print "nel corpus", file1, "il rapporto tra sostantivi e verbi è maggiore rispetto al corpus", file2, "\n"
	elif RapportoSV1 < RapportoSV2:
		print "nel corpus", file2, "il rapporto tra sostantivi e verbi è maggiore rispetto al corpus", file1, "\n"
	else:
		print "il rapporto tra sostantivi e verbi è uguale nei due corpora"
	print
	print "------------------------------"
	print
	print "6) La densità lessicale, calcolata come il rapporto tra il numero totale di occorrenze di Sostantivi, Verbi, Avverbi, Aggettivi e il numero totale di parole nel testo"
	print
	
	#calcolo la densità lessicale
	densitaLessicale1= ((numeroNomi1+numeroVerbi1+numeroAvverbi1+numeroAggettivi1)*1.0)/((lunghezza1-(numeroPunti1+numeroVirgole1))*1.0)
	print "nel corpus", file1, "è:", densitaLessicale1
	print
	densitaLessicale2= ((numeroNomi2+numeroVerbi2+numeroAvverbi2+numeroAggettivi2)*1.0)/((lunghezza2-(numeroPunti2+numeroVirgole2))*1.0)
	print "nel corpus", file2, "è:", densitaLessicale2
	print
	
	#confronto i due testi
	if densitaLessicale1 > densitaLessicale2:
		print "il corpus", file1, "ha una densità lessicale maggiore rispetto al corpus", file2, "\n"
	elif densitaLessicale1 < densitaLessicale2:
		print "il corpus", file2, "ha una densità lessicale maggiore rispetto al corpus", file1, "\n"
	else:
		print "i due corpora hanno la stessa densità lessicale"

main(sys.argv[1], sys.argv[2])