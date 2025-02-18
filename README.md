# neural-network

E' una rete neurale fatta "a mano", con la classica funzione di attivazione sigmoide.
Più nel dettaglio:
- Ho creato matrici di pesi che collegano i neuroni degli strati di input, lo strato nascosto e quello di output. Questi pesi sono inizializzati con valori casuali per permettere alla rete di apprendere.
- Ho implementato la funzione sigmoide, che mappa qualsiasi valore in un intervallo tra 0 e 1, e la sua derivata, utilizzata durante l'addestramento.
- Ho sviluppato il processo di feedforward, che passa gli input attraverso la rete per ottenere gli output. Questo permette alla rete di fare previsioni basate sui dati di input.  
- Ho implementato l'algoritmo di retropropagazione per addestrare la rete. Questo algoritmo minimizza l'errore aggiornando i pesi in base alla differenza tra gli output attesi e quelli effettivi.
