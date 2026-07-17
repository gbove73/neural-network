# Rete neurale "hand-made" per valutazione immobiliare

[![Java Version](https://img.shields.io/badge/Java-21%2B-blue.svg)](https://www.oracle.com/java/technologies/javase-jdk21-downloads.html)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Build](https://img.shields.io/badge/Build-Maven-red.svg)](https://maven.apache.org/)

## Descrizione del progetto

Questo repository contiene una rete neurale realizzata interamente in Java, senza TensorFlow, PyTorch o altre librerie di machine learning. Lo scopo è puramente didattico: rendere visibili i calcoli che una libreria professionale normalmente nasconde.

Il progetto usa la stima del prezzo di un immobile come esempio concreto. Non vuole sostituire una perizia o un modello addestrato su dati di mercato reali. Il dominio immobiliare serve a dare un significato intuitivo agli input e all’output:

1. superficie in metri quadrati;
2. numero di stanze;
3. numero di bagni;
4. piano;
5. valutazione della zona da 1 a 10;
6. prezzo stimato in euro come risultato.

Leggendo il codice è possibile seguire l’intero percorso di un dato: dalla normalizzazione iniziale, al forward pass, al calcolo dell’errore, alla backpropagation, fino all’aggiornamento di pesi e bias.

## Che cos’è una rete neurale?

Una rete neurale è un sistema composto da piccoli elementi di calcolo chiamati neuroni. Ogni neurone:

1. riceve alcuni numeri;
2. moltiplica ogni numero per un peso, che rappresenta l’importanza di quel collegamento;
3. somma i risultati;
4. aggiunge un bias, cioè una correzione indipendente dagli input;
5. applica una funzione di attivazione;
6. passa il risultato ai neuroni successivi.

Il termine “apprendimento” indica la modifica progressiva di pesi e bias per ridurre la distanza tra le previsioni e i risultati attesi.

Questa implementazione è un percettrone multistrato con:

- 5 neuroni di input;
- un singolo strato nascosto configurabile;
- 1 neurone di output;
- funzione di attivazione sigmoide;
- inizializzazione Xavier dei pesi;
- bias per lo strato nascosto e per quello di output;
- backpropagation;
- discesa stocastica del gradiente, o SGD;
- inverted dropout opzionale;
- seed configurabile per rendere gli esperimenti riproducibili.

## Il forward pass

Il forward pass è il viaggio dei dati dall’ingresso all’uscita.

Per ogni neurone viene prima calcolata una somma pesata:

```text
somma = bias + input₁ × peso₁ + input₂ × peso₂ + ... + inputₙ × pesoₙ
```

Alla somma viene applicata la sigmoide:

```text
sigmoide(x) = 1 / (1 + e⁻ˣ)
```

La sigmoide trasforma qualunque numero in un valore compreso tra 0 e 1. Questo introduce la non linearità necessaria per apprendere relazioni più complesse di una semplice retta.

Nel codice:

- `activateLayer` calcola somma pesata, bias e sigmoide;
- `forward` collega lo strato di input, quello nascosto e quello di output;
- `feedForward` esegue una previsione senza dropout.

## Errore e backpropagation

Dopo una previsione, la rete confronta il proprio risultato con quello atteso. Il progetto usa la loss quadratica:

```text
loss = (previsione - valore atteso)² / 2
```

Il quadrato rende positivi gli errori e dà maggiore peso agli errori grandi. Il fattore `1/2` semplifica la derivata.

La backpropagation risponde alla domanda: “quanto è responsabile ogni peso dell’errore finale?”. Procede dall’output verso gli input applicando la regola della catena.

Per il neurone di output:

```text
delta output =
    (previsione - valore atteso)
    × derivata della sigmoide
```

Per ogni neurone nascosto:

```text
delta nascosto =
    somma(delta output × peso verso output)
    × derivata della sigmoide nascosta
    × scala del dropout
```

Il gradiente di un collegamento è:

```text
gradiente del peso = valore ricevuto × delta del neurone di arrivo
```

Infine, gradient descent modifica il parametro:

```text
nuovo parametro =
    vecchio parametro - learning rate × gradiente
```

Il segno meno è importante: il gradiente indica la direzione in cui l’errore cresce più velocemente, quindi la rete si muove nella direzione opposta.

## Bias

Un bias è un parametro aggiunto alla somma pesata prima della funzione di attivazione. Può essere immaginato come una manopola che sposta la soglia di attivazione di un neurone.

Senza bias, un neurone sarebbe vincolato a una soglia centrata nello zero. Per esempio, con input tutti uguali a zero, la rete non potrebbe apprendere liberamente un risultato diverso usando soltanto i pesi.

Il progetto mantiene due vettori distinti:

- `hiddenBiases` per i neuroni nascosti;
- `outputBiases` per i neuroni di output.

I bias vengono appresi con la stessa regola usata per i pesi.

## Dropout

Il dropout è una tecnica di regolarizzazione. Durante il training spegne casualmente una percentuale dei neuroni nascosti, impedendo alla rete di dipendere sempre dagli stessi percorsi.

Questa implementazione usa inverted dropout:

1. crea una maschera casuale durante il forward pass di training;
2. assegna scala zero ai neuroni spenti;
3. amplifica i neuroni conservati con `1 / (1 - dropoutRate)`;
4. riusa la stessa maschera durante la backpropagation;
5. disattiva completamente il dropout durante la previsione.

Riutilizzare la stessa maschera è essenziale: un neurone spento durante la previsione di training non deve ricevere un gradiente come se avesse partecipato.

## Normalizzazione

Metri quadrati, numero di stanze e prezzi hanno ordini di grandezza molto diversi. Se fossero inviati direttamente alla rete, i valori grandi dominerebbero i calcoli.

La normalizzazione min-max usa:

```text
valore normalizzato = (valore - minimo) / (massimo - minimo)
```

Il minimo diventa 0, il massimo diventa 1 e un valore intermedio mantiene la propria posizione proporzionale.

Sono disponibili due modalità:

- range dichiarati esplicitamente, usati dalla configurazione dimostrativa;
- range appresi dal solo training set tramite `fit`.

Usare esclusivamente il training set evita il data leakage: il modello non deve conoscere informazioni ricavate dal test set prima della valutazione.

Per i valori fuori range sono disponibili tre strategie:

- `ALLOW`: continua la formula e consente risultati minori di 0 o maggiori di 1;
- `CLAMP`: limita il valore al minimo o al massimo più vicino;
- `REJECT`: segnala il dato come non valido.

## Installazione

Requisiti:

- JDK 21 o successivo;
- Maven 3.9 o successivo.

```bash
git clone https://github.com/gbove73/neural-network.git
cd neural-network
mvn verify
```

`mvn verify` compila il progetto, esegue tutti i test, genera il report JaCoCo e fallisce se la copertura di istruzioni o rami scende sotto il 100%.

Il report HTML viene generato in:

```text
target/site/jacoco/index.html
```

## Esempio di utilizzo

```java
import it.bove.application.RealEstateNeuralNetwork;
import it.bove.application.TrainingResult;
import it.bove.domain.realestate.PropertyFeatures;

RealEstateNeuralNetwork estimator = new RealEstateNeuralNetwork();

double[][] trainingProperties = {
    {80.0, 3.0, 1.0, 2.0, 7.0},
    {150.0, 4.0, 2.0, 3.0, 8.0},
    {50.0, 2.0, 1.0, 1.0, 5.0}
};

double[] trainingPrices = {
    220_000.0,
    380_000.0,
    150_000.0
};

TrainingResult result =
        estimator.train(trainingProperties, trainingPrices, 5_000);

System.out.println("Errore iniziale: " + result.initialMeanSquaredError());
System.out.println("Errore finale: " + result.finalMeanSquaredError());

PropertyFeatures property =
        new PropertyFeatures(100.0, 3, 1, 2, 6);

double estimatedPrice = estimator.estimatePrice(property);
System.out.println("Prezzo stimato: €" + Math.round(estimatedPrice));
```

L’overload originario rimane disponibile:

```java
double estimatedPrice =
        estimator.estimatePrice(100.0, 3, 1, 2, 6);
```

## Configurazione riproducibile

Una rete neurale contiene casualità nell’inizializzazione dei pesi e nel dropout. Un seed è il punto di partenza del generatore casuale: usando lo stesso seed si ottiene la stessa sequenza di numeri.

```java
NeuralNetworkConfiguration configuration =
        new NeuralNetworkConfiguration(
                5,      // input
                8,      // neuroni nascosti
                1,      // output
                0.05,   // learning rate
                0.10,   // dropout
                42L     // seed
        );

NeuralNetwork network = new NeuralNetwork(configuration);
```

Il seed non migliora il modello; rende l’esperimento ripetibile e quindi verificabile.

## Configurazione del training

Un’epoca è un giro completo su tutti gli esempi del training set. SGD aggiorna i parametri subito dopo ogni singolo esempio.

```java
TrainingConfiguration trainingConfiguration =
        new TrainingConfiguration(
                5_000,  // epoche
                true,   // mescola gli esempi a ogni epoca
                42L,    // seed dello shuffle
                500     // registra una metrica ogni 500 epoche
        );

TrainingResult result = estimator.train(
        trainingProperties,
        trainingPrices,
        trainingConfiguration
);
```

`TrainingResult` contiene:

- MSE iniziale;
- MSE finale;
- storico delle metriche registrate.

## Struttura del progetto

```text
it.bove
├── core
│   ├── nn
│   │   ├── NeuralNetwork
│   │   ├── NeuralNetworkConfiguration
│   │   └── NeuralNetworkModel
│   └── normalization
│       └── Normalizer
├── domain
│   └── realestate
│       ├── PropertyFeatures
│       ├── FeatureNormalizer
│       └── PriceNormalizer
├── application
│   ├── RealEstateNeuralNetwork
│   ├── TrainingConfiguration
│   ├── TrainingMetric
│   └── TrainingResult
└── infrastructure
    ├── nn
    │   └── NeuralNetworkAdapter
    └── normalization
        ├── DefaultFeatureNormalizer
        ├── DefaultPriceNormalizer
        └── OutOfRangePolicy
```

### Core

Contiene la matematica generale della rete, indipendente dal caso immobiliare.

### Domain

Dà nomi e regole ai dati immobiliari. `PropertyFeatures` evita di confondere la posizione delle cinque caratteristiche in un array anonimo.

### Application

Coordina normalizzazione, training, metriche, previsione e valutazione.

### Infrastructure

Contiene le implementazioni concrete dei normalizzatori e l’adapter che presenta `NeuralNetwork` attraverso l’interfaccia `NeuralNetworkModel`.

## Strategia di test

I test non si limitano a controllare che il programma “non lanci errori”. Verificano:

- valori noti di sigmoide e derivata;
- stabilità numerica con valori estremi;
- indipendenza degli array restituiti;
- apprendimento dei bias con input nulli;
- riproducibilità di pesi e dropout;
- gradient checking numerico;
- riduzione misurabile della loss;
- generalizzazione su immobili non presenti nel training set;
- separazione tra training e test;
- reversibilità della normalizzazione;
- tutte le politiche fuori range;
- validazione di configurazioni, dataset e valori di dominio;
- contratti dell’adapter;
- 100% delle istruzioni e dei rami secondo JaCoCo.

### Gradient checking

Il gradient checking confronta due modi indipendenti di calcolare la stessa pendenza:

1. la backpropagation produce il gradiente analitico;
2. il test aumenta e diminuisce ogni parametro di una quantità piccolissima;
3. osserva quanto cambia la loss;
4. ricava un gradiente numerico;
5. verifica che i due risultati coincidano entro una tolleranza.

Se le due pendenze coincidono, è molto improbabile che la formula della backpropagation contenga un errore di segno, una derivata mancante o un indice scambiato.

## Limiti dichiarati

Il progetto resta volutamente piccolo:

- supporta un solo strato nascosto;
- usa soltanto la sigmoide;
- esegue SGD su un esempio alla volta;
- non implementa mini-batch, ottimizzatori avanzati o accelerazione hardware;
- conserva pesi e bias soltanto in memoria;
- usa un dataset dimostrativo, non dati immobiliari reali;
- non produce una valutazione utilizzabile per decisioni economiche.

Questi limiti mantengono il codice leggibile e permettono di concentrarsi sui fondamenti matematici.

## Domande frequenti

### Perché otto neuroni nascosti?

Otto non è un numero matematicamente ottimale e non deriva da una regola universale. È una scelta didattica che offre una capacità sufficiente per l’esempio, mantenendo al tempo stesso dimensioni della rete e tempi di esecuzione contenuti.

In un progetto reale, il numero di neuroni sarebbe scelto confrontando più configurazioni su un validation set separato. Il test set verrebbe usato soltanto alla fine.

### Perché la sigmoide?

È semplice da visualizzare, ha una derivata compatta e rende chiara la regola della catena. Reti moderne usano spesso altre attivazioni, ma la sigmoide è adatta a un primo studio della backpropagation.

### Perché il modello può sbagliare anche se la loss diminuisce?

Ridurre l’errore sul training set significa adattarsi agli esempi osservati. Non garantisce automaticamente buone previsioni su casi nuovi. Per questo i test distinguono training set e test set e il README evita di presentare il progetto come uno stimatore professionale.

## Come contribuire

1. Crea un fork del repository.
2. Crea un branch descrittivo.
3. Mantieni commenti e Javadoc in italiano e identificatori in inglese.
4. Esegui `mvn verify`.
5. Verifica che test e copertura restino al 100%.
6. Aggiorna `CHANGELOG.md`.
7. Apri una pull request.

## Versione

La versione corrente è `1.0.0` e segue Semantic Versioning.

Le modifiche pubblicate e non ancora pubblicate sono documentate in `CHANGELOG.md`.

## Licenza

Il progetto è distribuito con licenza MIT. Consulta `LICENSE` per i dettagli.

## Contatti

Gianluca Bove - [@gbove73](https://github.com/gbove73)

Repository: [github.com/gbove73/neural-network](https://github.com/gbove73/neural-network)

---

*Questo progetto ha scopo esclusivamente dimostrativo ed educativo. Per sistemi reali si raccomandano dati rappresentativi, validazione rigorosa e librerie mature come TensorFlow, PyTorch o DeepLearning4J.*
