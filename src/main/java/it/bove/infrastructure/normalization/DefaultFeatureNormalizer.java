package it.bove.infrastructure.normalization;

import it.bove.domain.realestate.FeatureNormalizer;

import java.util.Arrays;

/**
 * Normalizzazione min-max delle caratteristiche immobiliari.
 *
 * <p>La formula è {@code (valore - minimo) / (massimo - minimo)}. Il minimo
 * diventa 0, il massimo diventa 1 e tutti i valori intermedi mantengono la
 * propria posizione proporzionale. Per esempio, 50 in un intervallo 0-100
 * diventa 0,5.</p>
 */
public final class DefaultFeatureNormalizer implements FeatureNormalizer {

    private final double[] featureMinimums;
    private final double[] featureMaximums;
    private final OutOfRangePolicy outOfRangePolicy;

    public DefaultFeatureNormalizer(double[] featureMinimums, double[] featureMaximums) {
        this(featureMinimums, featureMaximums, OutOfRangePolicy.ALLOW);
    }

    /**
     * Costruisce il normalizzatore proteggendo i range da modifiche esterne.
     */
    public DefaultFeatureNormalizer(
            double[] featureMinimums,
            double[] featureMaximums,
            OutOfRangePolicy outOfRangePolicy
    ) {
        validateRanges(featureMinimums, featureMaximums);
        if (outOfRangePolicy == null) {
            throw new IllegalArgumentException("La strategia fuori range è obbligatoria");
        }
        this.featureMinimums = Arrays.copyOf(featureMinimums, featureMinimums.length);
        this.featureMaximums = Arrays.copyOf(featureMaximums, featureMaximums.length);
        this.outOfRangePolicy = outOfRangePolicy;
    }

    /**
     * Apprende i range esclusivamente dal dataset fornito.
     */
    public static DefaultFeatureNormalizer fit(
            double[][] trainingFeatures,
            OutOfRangePolicy outOfRangePolicy
    ) {
        /*
         * "Fit" significa imparare i parametri della trasformazione dai dati.
         * Qui scorriamo ogni colonna: una colonna rappresenta sempre la stessa
         * caratteristica (per esempio i metri quadri), quindi ne cerchiamo
         * separatamente minimo e massimo.
         */
        validateDataset(trainingFeatures);
        int featureCount = trainingFeatures[0].length;
        double[] minimums = Arrays.copyOf(trainingFeatures[0], featureCount);
        double[] maximums = Arrays.copyOf(trainingFeatures[0], featureCount);
        for (double[] row : trainingFeatures) {
            if (row == null || row.length != featureCount) {
                throw new IllegalArgumentException("Tutte le righe devono avere la stessa dimensione");
            }
            for (int index = 0; index < featureCount; index++) {
                validateFinite(row[index]);
                minimums[index] = Math.min(minimums[index], row[index]);
                maximums[index] = Math.max(maximums[index], row[index]);
            }
        }
        ensureNonConstantRanges(minimums, maximums);
        return new DefaultFeatureNormalizer(minimums, maximums, outOfRangePolicy);
    }

    @Override
    public double[] normalize(double[] features) {
        validateVector(features, featureMinimums.length, "caratteristiche");
        double[] normalized = new double[features.length];
        for (int index = 0; index < features.length; index++) {
            double value = applyPolicy(features[index], featureMinimums[index], featureMaximums[index]);
            normalized[index] =
                    (value - featureMinimums[index]) / (featureMaximums[index] - featureMinimums[index]);
        }
        return normalized;
    }

    @Override
    public double[] denormalize(double[] normalizedFeatures) {
        validateVector(normalizedFeatures, featureMinimums.length, "caratteristiche normalizzate");
        double[] denormalized = new double[normalizedFeatures.length];
        for (int index = 0; index < normalizedFeatures.length; index++) {
            denormalized[index] = normalizedFeatures[index]
                    * (featureMaximums[index] - featureMinimums[index])
                    + featureMinimums[index];
        }
        return denormalized;
    }

    private double applyPolicy(double value, double minimum, double maximum) {
        return switch (outOfRangePolicy) {
            case ALLOW -> value;
            case CLAMP -> Math.clamp(value, minimum, maximum);
            case REJECT -> {
                if (value < minimum || value > maximum) {
                    throw new IllegalArgumentException("Caratteristica fuori dal range configurato");
                }
                yield value;
            }
        };
    }

    private static void validateRanges(double[] minimums, double[] maximums) {
        if (minimums == null || maximums == null || minimums.length == 0
                || minimums.length != maximums.length) {
            throw new IllegalArgumentException("I range devono esistere e avere la stessa dimensione");
        }
        for (int index = 0; index < minimums.length; index++) {
            validateFinite(minimums[index]);
            validateFinite(maximums[index]);
        }
        ensureNonConstantRanges(minimums, maximums);
    }

    private static void ensureNonConstantRanges(double[] minimums, double[] maximums) {
        for (int index = 0; index < minimums.length; index++) {
            if (minimums[index] >= maximums[index]) {
                throw new IllegalArgumentException("Ogni minimo deve essere minore del massimo corrispondente");
            }
        }
    }

    private static void validateDataset(double[][] values) {
        if (values == null || values.length == 0 || values[0] == null || values[0].length == 0) {
            throw new IllegalArgumentException("Il dataset non può essere vuoto");
        }
    }

    private static void validateVector(double[] values, int expectedLength, String label) {
        if (values == null || values.length != expectedLength) {
            throw new IllegalArgumentException("Dimensione non valida per " + label);
        }
        for (double value : values) {
            validateFinite(value);
        }
    }

    private static void validateFinite(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("I valori devono essere finiti");
        }
    }
}
