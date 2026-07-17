package it.bove.application;

import it.bove.core.nn.NeuralNetwork;
import it.bove.core.nn.NeuralNetworkConfiguration;
import it.bove.core.nn.NeuralNetworkModel;
import it.bove.domain.realestate.PropertyFeatures;
import it.bove.infrastructure.nn.NeuralNetworkAdapter;
import it.bove.infrastructure.normalization.DefaultFeatureNormalizer;
import it.bove.infrastructure.normalization.DefaultPriceNormalizer;
import it.bove.infrastructure.normalization.OutOfRangePolicy;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifica il caso d'uso immobiliare separando training set e test set.
 *
 * <p>Un test didattico deve chiarire non solo se il risultato è corretto, ma
 * anche quale proprietà del sistema sta proteggendo. Questa classe distingue:</p>
 *
 * <ul>
 *   <li>convergenza: la loss sul training set deve diminuire;</li>
 *   <li>generalizzazione: immobili mai mostrati devono ricevere stime ragionevoli;</li>
 *   <li>contratti: dati e dipendenze impossibili devono essere rifiutati subito;</li>
 *   <li>osservabilità: il chiamante deve poter leggere le metriche prodotte.</li>
 * </ul>
 */
class RealEstateNeuralNetworkTest {

    private static final double[][] TRAINING_FEATURES = {
            {50.0, 2.0, 1.0, 1.0, 3.0},
            {80.0, 3.0, 1.0, 2.0, 5.0},
            {110.0, 3.0, 2.0, 3.0, 6.0},
            {150.0, 4.0, 2.0, 4.0, 8.0},
            {200.0, 5.0, 3.0, 6.0, 9.0}
    };
    private static final double[] TRAINING_PRICES = {
            140_000.0,
            220_000.0,
            310_000.0,
            460_000.0,
            650_000.0
    };

    private RealEstateNeuralNetwork estimator;

    @BeforeEach
    void setUp() {
        NeuralNetwork network = new NeuralNetwork(
                new NeuralNetworkConfiguration(5, 8, 1, 0.15, 0.0, 42L)
        );
        estimator = new RealEstateNeuralNetwork(
                new NeuralNetworkAdapter(network),
                new DefaultFeatureNormalizer(
                        new double[]{30.0, 1.0, 1.0, 0.0, 1.0},
                        new double[]{250.0, 6.0, 4.0, 10.0, 10.0},
                        OutOfRangePolicy.REJECT
                ),
                new DefaultPriceNormalizer(
                        50_000.0,
                        900_000.0,
                        OutOfRangePolicy.REJECT
                )
        );
    }

    @Test
    void trainingReturnsMetricsAndReducesLoss() {
        /*
         * Non imponiamo un prezzo esatto: una rete apprende per approssimazione.
         * Verifichiamo invece un fatto più fondamentale e stabile, cioè che
         * l'errore finale sia molto più piccolo di quello osservato all'inizio.
         */
        TrainingResult result = estimator.train(
                TRAINING_FEATURES,
                TRAINING_PRICES,
                new TrainingConfiguration(4_000, true, 9L, 500)
        );

        assertTrue(result.finalMeanSquaredError() < result.initialMeanSquaredError() * 0.15);
        assertEquals(9, result.metrics().size());
    }

    @Test
    void trainingCanKeepTheOriginalOrderWhenShuffleIsDisabled() {
        /*
         * Lo shuffle è consigliato ma opzionale. Un solo giro senza shuffle
         * attraversa il ramo alternativo e dimostra che la configurazione non
         * obbliga il chiamante a cambiare l'ordine dei propri dati.
         */
        TrainingResult result = estimator.train(
                TRAINING_FEATURES,
                TRAINING_PRICES,
                new TrainingConfiguration(1, false, 9L, 1)
        );

        assertEquals(1, result.metrics().size());
    }

    @Test
    void modelGeneralizesToUnseenIntermediateProperties() {
        /*
         * I tre immobili qui sotto non compaiono nel training set. Sono valori
         * intermedi: chiediamo alla rete di interpolare la relazione appresa,
         * non di ricordare semplicemente una riga già vista.
         */
        estimator.train(TRAINING_FEATURES, TRAINING_PRICES, 6_000);
        double[][] testFeatures = {
                {65.0, 2.0, 1.0, 1.0, 4.0},
                {95.0, 3.0, 2.0, 3.0, 6.0},
                {175.0, 4.0, 3.0, 5.0, 9.0}
        };
        double[] testPrices = {180_000.0, 275_000.0, 550_000.0};

        assertTrue(estimator.evaluateModel(testFeatures, testPrices) < 20.0);
    }

    @Test
    void propertyValueObjectAndLegacyOverloadProduceSameEstimate() {
        /*
         * L'oggetto PropertyFeatures rende il codice più chiaro, ma l'overload
         * storico deve continuare a rappresentare esattamente gli stessi dati.
         */
        estimator.train(TRAINING_FEATURES, TRAINING_PRICES, 1_000);
        PropertyFeatures features = new PropertyFeatures(100.0, 3, 2, 3, 6);

        assertEquals(
                estimator.estimatePrice(features),
                estimator.estimatePrice(100.0, 3, 2, 3, 6),
                0.001
        );
    }

    @Test
    void invalidDatasetsAndPropertiesAreRejected() {
        /*
         * Questi casi non cercano una previsione: dimostrano che il confine
         * applicativo intercetta dati ambigui prima che raggiungano la matematica.
         */
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.train(new double[][]{{80.0, 3.0, 1.0, 2.0, 5.0}}, new double[0], 10)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.evaluateModel(
                        new double[][]{{80.0, 3.5, 1.0, 2.0, 5.0}},
                        new double[]{200_000.0}
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.estimatePrice(500.0, 3, 1, 2, 5)
        );
    }

    @Test
    void defaultAndFittedFactoriesCreateUsableEstimators() {
        /*
         * Il primo stimatore usa range documentati; il secondo impara minimo
         * e massimo dal solo training set. Verifichiamo entrambi i percorsi
         * pubblici perché rappresentano due modi diversi di preparare i dati.
         */
        RealEstateNeuralNetwork defaultEstimator = new RealEstateNeuralNetwork();
        RealEstateNeuralNetwork fittedEstimator =
                RealEstateNeuralNetwork.fittedTo(TRAINING_FEATURES, TRAINING_PRICES);

        assertTrue(Double.isFinite(defaultEstimator.estimatePrice(100.0, 3, 2, 3, 6)));
        fittedEstimator.train(TRAINING_FEATURES, TRAINING_PRICES, 10);
        assertTrue(Double.isFinite(fittedEstimator.estimatePrice(100.0, 3, 2, 3, 6)));
    }

    @Test
    void dependenciesTrainingConfigurationAndPropertyAreMandatory() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new RealEstateNeuralNetwork(null, null, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new RealEstateNeuralNetwork(
                        new FixedModel(new double[]{0.5}),
                        null,
                        new DefaultPriceNormalizer(50_000.0, 900_000.0)
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new RealEstateNeuralNetwork(
                        new FixedModel(new double[]{0.5}),
                        new DefaultFeatureNormalizer(
                                new double[]{0, 0, 0, 0, 0},
                                new double[]{1, 1, 1, 1, 1}
                        ),
                        null
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.train(TRAINING_FEATURES, TRAINING_PRICES, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.estimatePrice((PropertyFeatures) null)
        );
    }

    @Test
    void modelOutputMustContainExactlyOneFinitePrice() {
        /*
         * Il dominio prevede un solo prezzo. Usiamo un fake per simulare un
         * modello difettoso che restituisce zero, due o un output NaN.
         */
        RealEstateNeuralNetwork emptyOutputEstimator = estimatorWith(new double[0]);
        RealEstateNeuralNetwork nonFiniteOutputEstimator =
                estimatorWith(new double[]{Double.NaN});
        RealEstateNeuralNetwork wrongTrainingOutputEstimator =
                estimatorWith(new double[]{0.2, 0.3});

        assertThrows(
                IllegalStateException.class,
                () -> emptyOutputEstimator.estimatePrice(80.0, 3, 1, 2, 5)
        );
        assertThrows(
                IllegalStateException.class,
                () -> nonFiniteOutputEstimator.estimatePrice(80.0, 3, 1, 2, 5)
        );
        assertThrows(
                IllegalStateException.class,
                () -> wrongTrainingOutputEstimator.train(
                        new double[][]{{80.0, 3.0, 1.0, 2.0, 5.0}},
                        new double[]{200_000.0},
                        1
                )
        );
    }

    @Test
    void everyInvalidDatasetShapeOrPriceIsRejected() {
        /*
         * Un dataset è una relazione uno-a-uno: a ogni riga di caratteristiche
         * deve corrispondere un prezzo positivo e finito.
         */
        assertThrows(
                IllegalArgumentException.class,
                () -> RealEstateNeuralNetwork.fittedTo(null, TRAINING_PRICES)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> RealEstateNeuralNetwork.fittedTo(TRAINING_FEATURES, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> RealEstateNeuralNetwork.fittedTo(new double[0][], new double[0])
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.train(
                        new double[][]{{80.0, 3.0, 1.0, 2.0, 5.0}},
                        new double[]{Double.NaN},
                        1
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> estimator.train(
                        new double[][]{{80.0, 3.0, 1.0, 2.0, 5.0}},
                        new double[]{0.0},
                        1
                )
        );
    }

    @Test
    void invalidTrainingConfigurationsExplainTheirProblem() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingConfiguration(0, true, 1L, 1)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new TrainingConfiguration(1, true, 1L, 0)
        );
    }

    private RealEstateNeuralNetwork estimatorWith(double[] fixedOutput) {
        return new RealEstateNeuralNetwork(
                new FixedModel(fixedOutput),
                new DefaultFeatureNormalizer(
                        new double[]{30.0, 1.0, 1.0, 0.0, 1.0},
                        new double[]{250.0, 6.0, 4.0, 10.0, 10.0}
                ),
                new DefaultPriceNormalizer(50_000.0, 900_000.0)
        );
    }

    /**
     * Finto modello controllabile: non impara, ma restituisce sempre l'array
     * indicato dal test. Serve a provocare in modo deterministico risposte
     * impossibili che una rete reale difficilmente produrrebbe su comando.
     */
    private static final class FixedModel implements NeuralNetworkModel {

        private final double[] output;

        private FixedModel(double[] output) {
            this.output = output;
        }

        @Override
        public void train(double[] input, double[] expectedOutput) {
            // Nessuna operazione: questo fake isola il comportamento applicativo.
        }

        @Override
        public double[] predict(double[] input) {
            return output;
        }
    }
}
