package it.bove.application;

import it.bove.core.nn.NeuralNetwork;
import it.bove.core.nn.NeuralNetworkConfiguration;
import it.bove.core.nn.NeuralNetworkModel;
import it.bove.domain.realestate.FeatureNormalizer;
import it.bove.domain.realestate.PriceNormalizer;
import it.bove.domain.realestate.PropertyFeatures;
import it.bove.infrastructure.nn.NeuralNetworkAdapter;
import it.bove.infrastructure.normalization.DefaultFeatureNormalizer;
import it.bove.infrastructure.normalization.DefaultPriceNormalizer;
import it.bove.infrastructure.normalization.OutOfRangePolicy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Caso d'uso per addestrare e interrogare uno stimatore immobiliare dimostrativo.
 *
 * <p>Questa classe fa da ponte tra due mondi: da una parte gli immobili, descritti
 * con metri quadri, stanze e prezzo in euro; dall'altra la rete neurale, che sa
 * elaborare soltanto array di numeri in un intervallo piccolo. Il suo compito è
 * quindi validare i dati, normalizzarli, organizzare il training e tradurre la
 * previsione finale nuovamente in euro.</p>
 */
public final class RealEstateNeuralNetwork {

    private static final Logger LOGGER = LoggerFactory.getLogger(RealEstateNeuralNetwork.class);
    private static final int DEFAULT_HIDDEN_SIZE = 8;
    private static final double DEFAULT_LEARNING_RATE = 0.05;
    private static final double DEFAULT_DROPOUT_RATE = 0.1;
    private static final long DEFAULT_NETWORK_SEED = 42L;

    private final NeuralNetworkModel model;
    private final FeatureNormalizer featureNormalizer;
    private final PriceNormalizer priceNormalizer;

    /**
     * Costruttore per dependency injection e test isolati.
     */
    public RealEstateNeuralNetwork(
            NeuralNetworkModel model,
            FeatureNormalizer featureNormalizer,
            PriceNormalizer priceNormalizer
    ) {
        if (model == null || featureNormalizer == null || priceNormalizer == null) {
            throw new IllegalArgumentException("Modello e normalizzatori sono obbligatori");
        }
        this.model = model;
        this.featureNormalizer = featureNormalizer;
        this.priceNormalizer = priceNormalizer;
    }

    /**
     * Crea l'esempio predefinito con range dichiarati e seed riproducibile.
     */
    public RealEstateNeuralNetwork() {
        this(
                createDefaultModel(),
                new DefaultFeatureNormalizer(
                        new double[]{30.0, 1.0, 1.0, 0.0, 1.0},
                        new double[]{250.0, 5.0, 3.0, 10.0, 10.0},
                        OutOfRangePolicy.REJECT
                ),
                new DefaultPriceNormalizer(50_000.0, 900_000.0, OutOfRangePolicy.REJECT)
        );
    }

    /**
     * Crea uno stimatore i cui range derivano esclusivamente dal training set.
     *
     * <p>Il chiamante deve fornire qui soltanto dati di training, per evitare data leakage.</p>
     */
    public static RealEstateNeuralNetwork fittedTo(
            double[][] trainingFeatures,
            double[] trainingPrices
    ) {
        validateDataset(trainingFeatures, trainingPrices);
        return new RealEstateNeuralNetwork(
                createDefaultModel(),
                DefaultFeatureNormalizer.fit(trainingFeatures, OutOfRangePolicy.ALLOW),
                DefaultPriceNormalizer.fit(trainingPrices, OutOfRangePolicy.ALLOW)
        );
    }

    /**
     * Addestra il modello con le impostazioni predefinite.
     */
    public TrainingResult train(
            double[][] propertyFeatures,
            double[] propertyPrices,
            int epochs
    ) {
        return train(propertyFeatures, propertyPrices, TrainingConfiguration.defaults(epochs));
    }

    /**
     * Addestra con SGD, mescolando opzionalmente gli esempi a ogni epoca.
     */
    public TrainingResult train(
            double[][] propertyFeatures,
            double[] propertyPrices,
            TrainingConfiguration configuration
    ) {
        validateDataset(propertyFeatures, propertyPrices);
        if (configuration == null) {
            throw new IllegalArgumentException("La configurazione di training è obbligatoria");
        }

        /*
         * La sigmoide lavora meglio con valori confrontabili. Senza normalizzazione,
         * 200.000 euro dominerebbero numericamente valori come 3 stanze o 2 bagni.
         * Portiamo quindi ingressi e prezzi in una scala vicina a [0, 1].
         */
        double[][] normalizedFeatures = normalizeFeatures(propertyFeatures);
        double[][] normalizedPrices = normalizePrices(propertyPrices);

        // La loss iniziale è il punto di partenza con cui confronteremo il risultato.
        double initialLoss = calculateMeanSquaredError(normalizedFeatures, normalizedPrices);
        List<TrainingMetric> metrics = new ArrayList<>();
        int[] sampleOrder = createSequentialOrder(normalizedFeatures.length);
        RandomGenerator shuffleGenerator =
                RandomGeneratorFactory.getDefault().create(configuration.shuffleSeed());

        for (int epoch = 1; epoch <= configuration.epochs(); epoch++) {
            if (configuration.shuffleEachEpoch()) {
                /*
                 * Una epoca è un giro completo sul dataset. Cambiare l'ordine
                 * degli immobili evita che l'ultimo esempio visto influenzi sempre
                 * nello stesso modo l'aggiornamento successivo.
                 */
                shuffle(sampleOrder, shuffleGenerator);
            }
            for (int sampleIndex : sampleOrder) {
                // SGD aggiorna la rete subito dopo ogni singolo immobile.
                model.train(normalizedFeatures[sampleIndex], normalizedPrices[sampleIndex]);
            }
            if (shouldRecordMetrics(epoch, configuration)) {
                double loss = calculateMeanSquaredError(normalizedFeatures, normalizedPrices);
                metrics.add(new TrainingMetric(epoch, loss));
                LOGGER.info("Epoca {}: MSE={}", epoch, loss);
            }
        }

        double finalLoss = calculateMeanSquaredError(normalizedFeatures, normalizedPrices);
        LOGGER.info("Training completato: MSE iniziale={}, finale={}", initialLoss, finalLoss);
        return new TrainingResult(initialLoss, finalLoss, metrics);
    }

    /**
     * Stima il prezzo usando un valore di dominio esplicito.
     */
    public double estimatePrice(PropertyFeatures propertyFeatures) {
        if (propertyFeatures == null) {
            throw new IllegalArgumentException("Le caratteristiche dell'immobile sono obbligatorie");
        }
        double[] normalizedFeatures = featureNormalizer.normalize(propertyFeatures.toArray());
        double[] normalizedPrices = model.predict(normalizedFeatures);
        if (normalizedPrices.length != 1 || !Double.isFinite(normalizedPrices[0])) {
            throw new IllegalStateException("Il modello deve produrre un solo prezzo finito");
        }
        return priceNormalizer.denormalize(normalizedPrices[0]);
    }

    /**
     * Overload compatibile con l'API originaria.
     */
    public double estimatePrice(
            double squareMeters,
            int rooms,
            int bathrooms,
            int floor,
            int zoneRating
    ) {
        return estimatePrice(new PropertyFeatures(
                squareMeters,
                rooms,
                bathrooms,
                floor,
                zoneRating
        ));
    }

    /**
     * Calcola il MAPE su un dataset esterno al training.
     */
    public double evaluateModel(double[][] testFeatures, double[] testPrices) {
        validateDataset(testFeatures, testPrices);
        double percentageErrorSum = 0.0;
        for (int index = 0; index < testFeatures.length; index++) {
            PropertyFeatures propertyFeatures = PropertyFeatures.fromArray(testFeatures[index]);
            double estimatedPrice = estimatePrice(propertyFeatures);
            percentageErrorSum += Math.abs((estimatedPrice - testPrices[index]) / testPrices[index]);
        }
        return percentageErrorSum / testFeatures.length * 100.0;
    }

    private static NeuralNetworkModel createDefaultModel() {
        NeuralNetworkConfiguration configuration = new NeuralNetworkConfiguration(
                PropertyFeatures.FEATURE_COUNT,
                DEFAULT_HIDDEN_SIZE,
                1,
                DEFAULT_LEARNING_RATE,
                DEFAULT_DROPOUT_RATE,
                DEFAULT_NETWORK_SEED
        );
        return new NeuralNetworkAdapter(new NeuralNetwork(configuration));
    }

    private double[][] normalizeFeatures(double[][] features) {
        double[][] normalizedFeatures = new double[features.length][];
        for (int index = 0; index < features.length; index++) {
            // La conversione valida anche cardinalità e natura discreta dei campi.
            normalizedFeatures[index] =
                    featureNormalizer.normalize(PropertyFeatures.fromArray(features[index]).toArray());
        }
        return normalizedFeatures;
    }

    private double[][] normalizePrices(double[] prices) {
        double[][] normalizedPrices = new double[prices.length][1];
        for (int index = 0; index < prices.length; index++) {
            normalizedPrices[index][0] = priceNormalizer.normalize(prices[index]);
        }
        return normalizedPrices;
    }

    private double calculateMeanSquaredError(double[][] features, double[][] expectedOutputs) {
        /*
         * MSE significa Mean Squared Error, errore quadratico medio.
         * Elevare al quadrato rende positivi tutti gli errori e penalizza più
         * fortemente quelli grandi; dividere per il numero di output produce
         * una misura confrontabile tra dataset della stessa natura.
         */
        double squaredErrorSum = 0.0;
        int outputCount = 0;
        for (int sampleIndex = 0; sampleIndex < features.length; sampleIndex++) {
            double[] outputs = model.predict(features[sampleIndex]);
            if (outputs.length != expectedOutputs[sampleIndex].length) {
                throw new IllegalStateException("Dimensione dell'output del modello non valida");
            }
            for (int outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
                double error = outputs[outputIndex] - expectedOutputs[sampleIndex][outputIndex];
                squaredErrorSum += error * error;
                outputCount++;
            }
        }
        return squaredErrorSum / outputCount;
    }

    private static boolean shouldRecordMetrics(
            int epoch,
            TrainingConfiguration configuration
    ) {
        return epoch == 1
                || epoch == configuration.epochs()
                || epoch % configuration.metricsInterval() == 0;
    }

    private static int[] createSequentialOrder(int size) {
        int[] order = new int[size];
        for (int index = 0; index < size; index++) {
            order[index] = index;
        }
        return order;
    }

    private static void shuffle(int[] values, RandomGenerator randomGenerator) {
        for (int index = values.length - 1; index > 0; index--) {
            int replacementIndex = randomGenerator.nextInt(index + 1);
            int temporaryValue = values[index];
            values[index] = values[replacementIndex];
            values[replacementIndex] = temporaryValue;
        }
    }

    private static void validateDataset(double[][] features, double[] prices) {
        if (features == null || prices == null || features.length == 0) {
            throw new IllegalArgumentException("Il dataset non può essere vuoto");
        }
        if (features.length != prices.length) {
            throw new IllegalArgumentException("Ogni immobile deve avere un prezzo corrispondente");
        }
        for (int index = 0; index < features.length; index++) {
            PropertyFeatures.fromArray(features[index]);
            if (!Double.isFinite(prices[index]) || prices[index] <= 0.0) {
                throw new IllegalArgumentException("I prezzi devono essere positivi e finiti");
            }
        }
    }
}
