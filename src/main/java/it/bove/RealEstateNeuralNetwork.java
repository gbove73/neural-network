package it.bove;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sistema di valutazione immobiliare basato su reti neurali.
 * Implementa un modello predittivo per stimare i prezzi degli immobili
 * in base a caratteristiche come dimensione, posizione e qualità.
 */
public class RealEstateNeuralNetwork {
    // Utilizziamo un logger per tracciare l'esecuzione del programma e facilitare il debug
    private static final Logger log = LoggerFactory.getLogger(RealEstateNeuralNetwork.class);

    // Definiamo l'interfaccia per la rete neurale, seguendo il principio di Dependency Inversion
    private final NeuralNetworkModel model;

    // Definiamo i parametri per la normalizzazione dei dati di input
    private final FeatureNormalizer featureNormalizer;

    // Definiamo i parametri per la normalizzazione dei prezzi
    private final PriceNormalizer priceNormalizer;

    /**
     * Costruttore che accetta un modello di rete neurale e i normalizzatori.
     * Seguendo il principio di Dependency Injection, riceviamo le dipendenze dall'esterno.
     *
     * @param model             Il modello di rete neurale da utilizzare
     * @param featureNormalizer Normalizzatore per le caratteristiche degli immobili
     * @param priceNormalizer   Normalizzatore per i prezzi degli immobili
     */
    public RealEstateNeuralNetwork(
            NeuralNetworkModel model,
            FeatureNormalizer featureNormalizer,
            PriceNormalizer priceNormalizer) {
        // Inizializziamo il modello con le dipendenze fornite dall'esterno
        this.model = model;
        this.featureNormalizer = featureNormalizer;
        this.priceNormalizer = priceNormalizer;

        // Registriamo l'inizializzazione nel log per tracciabilità
        log.info("Sistema di valutazione immobiliare inizializzato con modello: {}", model.getClass().getSimpleName());
    }

    /**
     * Costruttore di convenienza che crea un sistema con configurazione predefinita.
     * Utilizza una rete neurale standard e parametri di normalizzazione comunemente usati.
     */
    public RealEstateNeuralNetwork() {
        // Creiamo una rete neurale standard: 5 input (caratteristiche), 8 neuroni nascosti, 1 output (prezzo)
        this.model = new NeuralNetworkAdapter(new NeuralNetwork(5, 8, 1, 0.05, 0.1));

        // Definiamo i parametri di normalizzazione basati su analisi statistiche del mercato immobiliare
        this.featureNormalizer = new DefaultFeatureNormalizer(
                new double[]{30.0, 1.0, 1.0, 0.0, 1.0},   // Valori minimi: mq, stanze, bagni, piano, zona
                new double[]{250.0, 5.0, 3.0, 10.0, 10.0} // Valori massimi
        );

        // Definiamo i parametri di normalizzazione dei prezzi basati sul mercato locale
        this.priceNormalizer = new DefaultPriceNormalizer(50000.0, 900000.0);

        // Registriamo l'inizializzazione con configurazione standard
        log.info("Sistema di valutazione immobiliare inizializzato con configurazione predefinita");
    }

    /**
     * Addestra il modello con un dataset di immobili e i relativi prezzi.
     *
     * @param propertyFeatures Matrice delle caratteristiche degli immobili di training
     * @param propertyPrices   Array dei prezzi corrispondenti
     * @param epochs           Numero di epoche di addestramento
     */
    public void train(double[][] propertyFeatures, double[] propertyPrices, int epochs) {
        // Registriamo l'inizio del processo di addestramento
        log.info("Inizio addestramento con {} immobili per {} epoche", propertyFeatures.length, epochs);

        // Normalizziamo i dati prima dell'addestramento per migliorare le performance della rete neurale
        double[][] normalizedFeatures = normalizeFeatures(propertyFeatures);
        double[][] normalizedPrices = normalizePrices(propertyPrices);

        // Registriamo l'errore iniziale prima dell'addestramento per monitorare il miglioramento
        double initialError = calculateAverageError(normalizedFeatures, normalizedPrices);
        log.info("Errore iniziale prima dell'addestramento: {}", initialError);

        // Addestriamo il modello per il numero specificato di epoche
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Per ciascuna epoca, processiamo tutti gli esempi di addestramento
            for (int i = 0; i < normalizedFeatures.length; i++) {
                // Addestriamo la rete con un esempio alla volta, seguendo l'algoritmo di discesa del gradiente stocastico
                model.train(normalizedFeatures[i], normalizedPrices[i]);
            }

            // Ogni 1000 epoche, registriamo l'errore corrente per monitorare il progresso
            if (epoch % 1000 == 0 || epoch == epochs - 1) {
                double currentError = calculateAverageError(normalizedFeatures, normalizedPrices);
                log.info("Epoca {}: errore = {}", epoch, currentError);
            }
        }

        // Calcoliamo e registriamo l'errore finale per verificare il successo dell'addestramento
        double finalError = calculateAverageError(normalizedFeatures, normalizedPrices);
        log.info("Addestramento completato. Errore finale: {}", finalError);
    }

    /**
     * Stima il prezzo di un immobile in base alle sue caratteristiche.
     *
     * @param squareMeters Superficie dell'immobile in metri quadri
     * @param rooms        Numero di stanze
     * @param bathrooms    Numero di bagni
     * @param floor        Piano dell'immobile (0 = piano terra)
     * @param zoneRating   Valutazione della zona (1-10, dove 10 è la zona migliore)
     * @return Prezzo stimato dell'immobile in euro
     */
    public double estimatePrice(double squareMeters, int rooms, int bathrooms, int floor, int zoneRating) {
        // Creiamo un array con le caratteristiche dell'immobile da valutare
        double[] features = {squareMeters, rooms, bathrooms, floor, zoneRating};

        // Registriamo i dettagli dell'immobile per tracciabilità
        log.info("Valutazione immobile: {}mq, {} stanze, {} bagni, piano {}, zona {}",
                squareMeters, rooms, bathrooms, floor, zoneRating);

        // Normalizziamo le caratteristiche per l'input alla rete neurale
        double[] normalizedFeatures = featureNormalizer.normalize(features);

        // Utilizziamo il modello per predire il prezzo normalizzato
        double[] normalizedPriceArray = model.predict(normalizedFeatures);
        double normalizedPrice = normalizedPriceArray[0]; // Prendiamo l'unico valore di output

        // Denormalizziamo il prezzo per ottenere il valore in euro
        double estimatedPrice = priceNormalizer.denormalize(normalizedPrice);

        // Registriamo il risultato della valutazione
        log.info("Prezzo stimato: €{}", (int) estimatedPrice);

        // Restituiamo il prezzo stimato
        return estimatedPrice;
    }

    /**
     * Valuta l'accuratezza del modello su un set di dati di test.
     *
     * @param testFeatures Caratteristiche degli immobili di test
     * @param testPrices   Prezzi reali degli immobili di test
     * @return Errore medio assoluto percentuale (MAPE)
     */
    public double evaluateModel(double[][] testFeatures, double[] testPrices) {
        // Registriamo l'inizio della valutazione
        log.info("Valutazione del modello su {} immobili", testFeatures.length);

        // Calcoliamo la somma degli errori percentuali assoluti
        double sumPercentageError = 0.0;

        // Per ogni immobile nel dataset di test
        for (int i = 0; i < testFeatures.length; i++) {
            // Estraiamo le caratteristiche dell'immobile corrente
            double squareMeters = testFeatures[i][0];
            int rooms = (int) testFeatures[i][1];
            int bathrooms = (int) testFeatures[i][2];
            int floor = (int) testFeatures[i][3];
            int zoneRating = (int) testFeatures[i][4];

            // Stimiamo il prezzo usando il nostro modello
            double estimatedPrice = estimatePrice(squareMeters, rooms, bathrooms, floor, zoneRating);

            // Calcoliamo l'errore percentuale assoluto
            double realPrice = testPrices[i];
            double percentageError = Math.abs((estimatedPrice - realPrice) / realPrice);

            // Sommiamo l'errore al totale
            sumPercentageError += percentageError;

            // Registriamo l'errore per questo immobile specifico
            log.debug("Immobile {}: Reale €{}, Stimato €{}, Errore {}%",
                    i, (int) realPrice, (int) estimatedPrice, (int) (percentageError * 100));
        }

        // Calcoliamo l'errore medio percentuale assoluto (MAPE)
        double mape = sumPercentageError / testFeatures.length * 100; // in percentuale

        // Registriamo il risultato della valutazione
        log.info("Errore medio percentuale assoluto (MAPE): {}%", (int) mape);

        // Restituiamo il MAPE
        return mape;
    }

    /**
     * Normalizza le caratteristiche di tutti gli immobili nel dataset.
     *
     * @param features Matrice delle caratteristiche degli immobili
     * @return Matrice delle caratteristiche normalizzate
     */
    private double[][] normalizeFeatures(double[][] features) {
        // Creiamo una matrice per i valori normalizzati
        double[][] normalized = new double[features.length][];

        // Normalizziamo ogni riga (immobile) usando il nostro normalizzatore
        for (int i = 0; i < features.length; i++) {
            normalized[i] = featureNormalizer.normalize(features[i]);
        }

        // Restituiamo la matrice normalizzata
        return normalized;
    }

    /**
     * Normalizza i prezzi degli immobili nel dataset.
     *
     * @param prices Array dei prezzi degli immobili
     * @return Matrice dei prezzi normalizzati in formato compatibile col modello
     */
    private double[][] normalizePrices(double[] prices) {
        // Creiamo una matrice di output, dove ogni riga ha un solo valore (il prezzo normalizzato)
        double[][] normalized = new double[prices.length][1];

        // Normalizziamo ogni prezzo usando il nostro normalizzatore
        for (int i = 0; i < prices.length; i++) {
            normalized[i][0] = priceNormalizer.normalize(prices[i]);
        }

        // Restituiamo la matrice di prezzi normalizzati
        return normalized;
    }

    /**
     * Calcola l'errore medio sui dati normalizzati.
     *
     * @param features        Caratteristiche normalizzate
     * @param expectedOutputs Output attesi normalizzati
     * @return Errore medio
     */
    private double calculateAverageError(double[][] features, double[][] expectedOutputs) {
        // Inizializziamo la somma degli errori
        double errorSum = 0;

        // Per ogni esempio nel dataset
        for (int i = 0; i < features.length; i++) {
            // Prediciamo l'output usando il modello
            double[] output = model.predict(features[i]);

            // Per ogni valore di output (nel nostro caso è solo uno - il prezzo)
            for (int j = 0; j < output.length; j++) {
                // Sommiamo l'errore assoluto tra previsione e valore atteso
                errorSum += Math.abs(expectedOutputs[i][j] - output[j]);
            }
        }

        // Restituiamo l'errore medio
        return errorSum / features.length;
    }

    /**
     * Interfaccia che definisce il comportamento di un modello di rete neurale.
     * Seguendo il principio di Interface Segregation, definiamo un'interfaccia minimale.
     */
    public interface NeuralNetworkModel {
        /**
         * Addestra il modello con un esempio.
         *
         * @param input          Caratteristiche di input normalizzate
         * @param expectedOutput Output atteso normalizzato
         */
        void train(double[] input, double[] expectedOutput);

        /**
         * Esegue una predizione.
         *
         * @param input Caratteristiche di input normalizzate
         * @return Output previsto normalizzato
         */
        double[] predict(double[] input);
    }

    /**
     * Adapter per la classe NeuralNetwork esistente.
     * Seguendo il pattern Adapter, adattiamo l'implementazione esistente alla nostra interfaccia.
     */
    private static class NeuralNetworkAdapter implements NeuralNetworkModel {
        // La rete neurale concreta da adattare
        private final NeuralNetwork neuralNetwork;

        /**
         * Costruttore che accetta una rete neurale da adattare.
         *
         * @param neuralNetwork La rete neurale da adattare
         */
        public NeuralNetworkAdapter(NeuralNetwork neuralNetwork) {
            // Memorizziamo la rete neurale da adattare
            this.neuralNetwork = neuralNetwork;
        }

        @Override
        public void train(double[] input, double[] expectedOutput) {
            // Deleghiamo l'addestramento alla rete neurale concreta
            neuralNetwork.train(input, expectedOutput);
        }

        @Override
        public double[] predict(double[] input) {
            // Deleghiamo la predizione alla rete neurale concreta
            return neuralNetwork.feedForward(input);
        }
    }

    /**
     * Interfaccia per la normalizzazione delle caratteristiche degli immobili.
     * Seguendo il principio di Single Responsibility, separiamo la logica di normalizzazione.
     */
    public interface FeatureNormalizer {
        /**
         * Normalizza le caratteristiche di un immobile.
         *
         * @param features Array delle caratteristiche da normalizzare
         * @return Array delle caratteristiche normalizzate
         */
        double[] normalize(double[] features);
    }

    /**
     * Implementazione predefinita del normalizzatore di caratteristiche.
     */
    private static class DefaultFeatureNormalizer implements FeatureNormalizer {
        // Valori minimi per ciascuna caratteristica
        private final double[] featureMin;
        // Valori massimi per ciascuna caratteristica
        private final double[] featureMax;

        /**
         * Costruttore con valori minimi e massimi per la normalizzazione.
         *
         * @param featureMin Array dei valori minimi per ciascuna caratteristica
         * @param featureMax Array dei valori massimi per ciascuna caratteristica
         */
        public DefaultFeatureNormalizer(double[] featureMin, double[] featureMax) {
            // Memorizziamo i valori minimi e massimi
            this.featureMin = featureMin;
            this.featureMax = featureMax;
        }

        @Override
        public double[] normalize(double[] features) {
            // Creiamo un nuovo array per i valori normalizzati
            double[] normalized = new double[features.length];

            // Per ogni caratteristica, applichiamo la formula di normalizzazione
            for (int i = 0; i < features.length; i++) {
                // Formula: (valore - minimo) / (massimo - minimo)
                normalized[i] = (features[i] - featureMin[i]) / (featureMax[i] - featureMin[i]);
            }

            // Restituiamo l'array normalizzato
            return normalized;
        }
    }

    /**
     * Interfaccia per la normalizzazione dei prezzi degli immobili.
     * Seguendo il principio di Interface Segregation, creiamo un'interfaccia specifica.
     */
    public interface PriceNormalizer {
        /**
         * Normalizza un prezzo.
         *
         * @param price Prezzo da normalizzare
         * @return Prezzo normalizzato nell'intervallo [0,1]
         */
        double normalize(double price);

        /**
         * Denormalizza un prezzo normalizzato.
         *
         * @param normalizedPrice Prezzo normalizzato nell'intervallo [0,1]
         * @return Prezzo reale in euro
         */
        double denormalize(double normalizedPrice);
    }

    /**
     * Implementazione predefinita del normalizzatore di prezzi.
     */
    private static class DefaultPriceNormalizer implements PriceNormalizer {
        // Prezzo minimo nel dataset
        private final double priceMin;
        // Prezzo massimo nel dataset
        private final double priceMax;

        /**
         * Costruttore con prezzi minimi e massimi per la normalizzazione.
         *
         * @param priceMin Prezzo minimo in euro
         * @param priceMax Prezzo massimo in euro
         */
        public DefaultPriceNormalizer(double priceMin, double priceMax) {
            // Memorizziamo i prezzi minimi e massimi
            this.priceMin = priceMin;
            this.priceMax = priceMax;
        }

        @Override
        public double normalize(double price) {
            // Applichiamo la formula di normalizzazione: (prezzo - minimo) / (massimo - minimo)
            return (price - priceMin) / (priceMax - priceMin);
        }

        @Override
        public double denormalize(double normalizedPrice) {
            // Applichiamo la formula inversa: normalizzato * (massimo - minimo) + minimo
            return normalizedPrice * (priceMax - priceMin) + priceMin;
        }
    }
}