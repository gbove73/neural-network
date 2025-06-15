# Rete neurale "hand-made" per valutazione immobiliare

[![Java Version](https://img.shields.io/badge/Java-21%2B-blue.svg)](https://www.oracle.com/java/technologies/javase-jdk21-downloads.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/badge/Build-Maven-red.svg)](https://maven.apache.org/)

## 📋 Descrizione del progetto

Questo repository contiene un progetto **puramente dimostrativo** di una rete neurale implementata completamente a mano in Java, senza utilizzare framework o librerie di machine learning. L'obiettivo principale è didattico: mostrare i fondamenti dell'implementazione di una rete neurale partendo dai primi principi matematici.

La rete neurale è stata applicata al caso d'uso della valutazione immobiliare come esempio pratico, ma il focus è sulla comprensione dell'algoritmo di backpropagation e del funzionamento interno delle reti neurali artificiali.

## 🧠 Perché una rete neurale scritta a mano?

Mentre esistono numerose librerie mature per il machine learning come TensorFlow, PyTorch o DeepLearning4J, questo progetto evita deliberatamente di utilizzarle per:

- **Scopo didattico** - Comprendere pienamente il funzionamento interno delle reti neurali
- **Trasparenza algortimica** - Visualizzare esattamente cosa accade durante l'addestramento e l'inferenza
- **Controllo completo** - Implementare ogni aspetto dell'algoritmo senza astrazioni
- **Semplicità** - Mantenere il codice leggibile e comprensibile senza dipendenze complesse

## 🏗️ Architettura della rete neurale

La rete implementata è un percettrone multistrato con:

- **Input Layer** - 5 neuroni rappresentanti le caratteristiche immobiliari
- **Hidden Layer** - Singolo strato nascosto configurabile (default: 8 neuroni)
- **Output Layer** - Singolo neurone che produce la stima del prezzo
- **Funzione di attivazione** - Sigmoide per l'introduzione di non-linearità
- **Addestramento** - Backpropagation con discesa stocastica del gradiente
- **Regolarizzazione** - Implementazione del dropout per prevenire l'overfitting

## 🚀 Installazione

```bash
# Clona il repository
git clone git@github.com:gbove73/neural-network.git

# Entra nella directory del progetto
cd neural-network

# Compila il progetto con Maven
mvn clean install
```

## 💻 Esempio di Utilizzo

```java
// Crea una rete neurale con la configurazione predefinita
RealEstateNeuralNetwork evaluator = new RealEstateNeuralNetwork();

// Definisce un semplice dataset di addestramento
double[][] properties = {
    {80.0, 3.0, 1.0, 2.0, 7.0},   // 220.000€ - appartamento medio in buona zona
    {150.0, 4.0, 2.0, 3.0, 8.0},  // 380.000€ - appartamento grande in ottima zona
    {50.0, 2.0, 1.0, 1.0, 5.0}    // 150.000€ - appartamento piccolo in zona media
};
double[] prices = {220000.0, 380000.0, 150000.0};

// Addestra il modello per 5000 epoche
evaluator.train(properties, prices, 5000);

// Stima il prezzo di un nuovo immobile
double price = evaluator.estimatePrice(100.0, 3, 1, 2, 6);
System.out.println("Prezzo stimato: €" + (int)price);
```

## 📚 Struttura del progetto

### 🏛️ Organizzazione dei package

Il progetto è strutturato seguendo i principi di Clean Architecture, che garantisce separazione delle responsabilità e indipendenza dai framework:

```
it.bove
├── core                 // Logica di business centrale indipendente dal dominio
│   ├── nn               // Implementazione base rete neurale
│   └── normalization    // Normalizzazione dati generica
├── domain               // Regole di business specifiche del dominio
│   └── realestate       // Dominio della valutazione immobiliare
├── application          // Casi d'uso dell'applicazione
└── infrastructure       // Adattatori e implementazioni concrete
```

#### Strati Architetturali

- **Core**: Contiene la logica base indipendente dal dominio
  - `NeuralNetwork` - Implementazione matematica della rete neurale
  - `NeuralNetworkModel` - Interfaccia per modelli di rete neurale
  - `Normalizer<T,R>` - Interfaccia generica per normalizzazione dei dati

- **Domain**: Contiene le regole di business e interfacce specifiche del dominio
  - `PriceNormalizer` - Interfaccia per normalizzazione prezzi immobiliari
  - `FeatureNormalizer` - Interfaccia per normalizzazione caratteristiche immobiliari

- **Application**: Implementa i casi d'uso dell'applicazione
  - `RealEstateNeuralNetwork` - Sistema di valutazione immobiliare

- **Infrastructure**: Contiene implementazioni concrete delle interfacce
  - `NeuralNetworkAdapter` - Adapter per connettere la rete neurale all'interfaccia del modello
  - `DefaultPriceNormalizer` - Implementazione concreta per normalizzazione prezzi
  - `DefaultFeatureNormalizer` - Implementazione concreta per normalizzazione caratteristiche

#### Vantaggi dell'architettura

- **Indipendenza dai Framework** - Il core e il dominio non dipendono da librerie esterne
- **Testabilità** - Le interfacce permettono di testare i componenti in isolamento
- **Flessibilità** - Facile sostituire implementazioni (es. diversa strategia di normalizzazione)
- **Manutenibilità** - Ogni componente ha una responsabilità chiara e ben definita

### `NeuralNetwork.java`

Il cuore del progetto: implementazione da zero di una rete neurale feedforward con:

- **Inizializzazione dei pesi** - Valori casuali per i collegamenti tra neuroni
- **Feedforward** - Propagazione del segnale attraverso la rete
- **Backpropagation** - Calcolo dell'errore e aggiustamento dei pesi
- **Dropout** - Tecnica per prevenire l'overfitting
- **Funzioni di attivazione** - Implementazioni di Sigmoide e ReLU

### `RealEstateNeuralNetwork.java`

Wrapper che applica la rete neurale al contesto immobiliare:

- **Normalizzazione** - Preprocessamento dei dati per la rete neurale
- **Pattern Adapter** - Integrazione modulare con la rete neurale
- **Valutazione** - Metriche per misurare l'accuratezza delle predizioni

### `RealEstateNeuralNetworkTest.java`

Test completi che verificano:

- **Convergenza** - Diminuzione dell'errore durante l'addestramento
- **Accuratezza** - Capacità predittiva su esempi noti
- **Generalizzazione** - Comportamento con dati mai visti
- **Robustezza** - Reazione a scenari limite

### `DefaultFeatureNormalizer.java`

Implementazione predefinita del normalizzatore di caratteristiche:

- **Normalizzazione** - Trasformazione delle caratteristiche degli immobili in un intervallo normalizzato
- **Denormalizzazione** - Riconversione delle caratteristiche normalizzate ai valori originali

### `DefaultPriceNormalizer.java`

Implementazione predefinita del normalizzatore di prezzi:

- **Normalizzazione** - Trasformazione dei prezzi degli immobili in un intervallo normalizzato
- **Denormalizzazione** - Riconversione dei prezzi normalizzati ai valori originali

### `FeatureNormalizer.java`

Interfaccia per la normalizzazione delle caratteristiche degli immobili:

- **Normalizzazione** - Definisce il metodo per normalizzare le caratteristiche
- **Denormalizzazione** - Definisce il metodo per denormalizzare le caratteristiche

### `PriceNormalizer.java`

Interfaccia per la normalizzazione dei prezzi degli immobili:

- **Normalizzazione** - Definisce il metodo per normalizzare i prezzi
- **Denormalizzazione** - Definisce il metodo per denormalizzare i prezzi

### `NeuralNetworkAdapter.java`

Adapter per la classe `NeuralNetwork` esistente:

- **Train** - Addestra la rete neurale con un esempio
- **Predict** - Esegue una predizione utilizzando la rete neurale

### `NeuralNetworkModel.java`

Interfaccia che definisce il comportamento di un modello di rete neurale:

- **Train** - Addestra il modello con un esempio
- **Predict** - Esegue una predizione utilizzando il modello

### `Normalizer.java`

Interfaccia generica per la normalizzazione dei dati:

- **Normalizzazione** - Definisce il metodo per normalizzare un valore
- **Denormalizzazione** - Definisce il metodo per denormalizzare un valore

## 🛠️ Requisiti Tecnici

- **Java 21+** - Utilizzo delle funzionalità più recenti del linguaggio
- **Maven** - Gestione delle dipendenze e build automatizzata
- **SLF4J** - Logging strutturato per debug e tracciabilità
- **JUnit 5** - Framework per i test automatizzati

## ⚠️ Limitazioni

Essendo un progetto dimostrativo, presenta alcune limitazioni:

- **Performance** - Non ottimizzato per grandi dataset (usa calcoli naïf)
- **Funzioni di attivazione** - Supporta Sigmoide e ReLU
- **Architettura** - Supporta solo una topologia fissa con un singolo strato nascosto
- **Batch processing** - Non implementa il mini-batch gradient descent

## 🤝 Come Contribuire

Questo è un progetto didattico, ma i contributi sono benvenuti:

1. Fork del repository
2. Crea un branch (`git checkout -b feature/miglioramento-xyz`)
3. Commit delle modifiche (`git commit -m 'Aggiunto xyz'`)
4. Push al branch (`git push origin feature/miglioramento-xyz`)
5. Apri una Pull Request

## 📝 Licenza

Questo progetto è rilasciato sotto licenza MIT. Consulta il file `LICENSE` per maggiori dettagli.

## F.A.Q.

#### Perché hai scelto proprio 8 neuroni per l’hidden layer?

Il numero di neuroni dovrebbe seguire la "rule of thumb" con un valore tra il numero di neuroni di input - in questo
caso 5, le caratteristiche dell'immobile - e il numero di neuroni di output - in questo caso 1, il prezzo dell'immobile.
Ne ho scelti 8 per sperimentare l'overfitting, rischio che si corre con troppi neuroni nello strato hidden, e
verificarne il meccanismo di dropout implementato per mitigarlo. Con troppi neuroni per lo strato nascosto il rischio è
che la rete neurale si adatti troppo ai dati di addestramento, e quindi il modello impara troppo bene i dettagli e il "
rumore" del set di dati di addestramento perdendo la capacità di generalizzare su nuovi dati non visti.

## 📞 Contatti

Gianluca Bove - [@gbove73](https://github.com/gbove73)

Repository: [github.com/gbove73/neural-network](https://github.com/gbove73/neural-network)

---

*Questo progetto ha scopo puramente dimostrativo ed educativo. L'implementazione manuale di una rete neurale è un esercizio didattico: in ambito produttivo si consiglia di utilizzare librerie mature e ottimizzate come TensorFlow, PyTorch o DeepLearning4J.*