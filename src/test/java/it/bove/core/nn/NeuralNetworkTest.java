package it.bove.core.nn;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifica sia il contratto pubblico sia la correttezza matematica della rete.
 */
class NeuralNetworkTest {

    private static final double TOLERANCE = 1.0e-6;

    @Test
    void sigmoidAndDerivativeMatchKnownValues() {
        NeuralNetwork network = createNetwork(0.0, 7L);

        assertEquals(0.5, network.sigmoid(0.0), TOLERANCE);
        assertEquals(0.25, network.sigmoidDerivative(0.5), TOLERANCE);
        assertTrue(Double.isFinite(network.sigmoid(1_000.0)));
        assertTrue(Double.isFinite(network.sigmoid(-1_000.0)));
    }

    @Test
    void feedForwardReturnsIndependentResults() {
        NeuralNetwork network = createNetwork(0.0, 7L);
        double[] firstPrediction = network.feedForward(new double[]{0.2, 0.8});
        double originalValue = firstPrediction[0];

        firstPrediction[0] = -1.0;
        double[] secondPrediction = network.feedForward(new double[]{0.2, 0.8});

        assertEquals(originalValue, secondPrediction[0], TOLERANCE);
        assertNotEquals(firstPrediction[0], secondPrediction[0]);
    }

    @Test
    void analyticGradientMatchesFiniteDifferences() {
        NeuralNetwork network = createNetwork(0.0, 19L);
        double[] inputs = {0.25, 0.75};
        double[] expectedOutputs = {0.6};
        double[] originalParameters = network.parametersSnapshot();
        double[] analyticGradient = network.gradientSnapshot(inputs, expectedOutputs);
        double[] numericGradient = new double[originalParameters.length];
        double epsilon = 1.0e-6;

        /*
         * Il gradient checking perturba ogni parametro di una quantità piccolissima.
         * La pendenza numerica (loss destra - loss sinistra) / 2ε deve coincidere
         * con quella calcolata dalla backpropagation.
         */
        for (int index = 0; index < originalParameters.length; index++) {
            double[] positiveParameters = originalParameters.clone();
            positiveParameters[index] += epsilon;
            network.restoreParameters(positiveParameters);
            double positiveLoss = network.calculateLoss(inputs, expectedOutputs);

            double[] negativeParameters = originalParameters.clone();
            negativeParameters[index] -= epsilon;
            network.restoreParameters(negativeParameters);
            double negativeLoss = network.calculateLoss(inputs, expectedOutputs);

            numericGradient[index] = (positiveLoss - negativeLoss) / (2.0 * epsilon);
        }
        network.restoreParameters(originalParameters);

        assertArrayEquals(numericGradient, analyticGradient, 1.0e-5);
    }

    @Test
    void trainingReducesLossEvenWithZeroInputsBecauseBiasesAreLearned() {
        NeuralNetwork network = createNetwork(0.0, 31L);
        double[] inputs = {0.0, 0.0};
        double[] expectedOutputs = {0.9};
        double initialLoss = network.calculateLoss(inputs, expectedOutputs);

        for (int iteration = 0; iteration < 2_000; iteration++) {
            network.train(inputs, expectedOutputs);
        }

        assertTrue(network.calculateLoss(inputs, expectedOutputs) < initialLoss * 0.05);
    }

    @Test
    void sameSeedMakesInitializationAndDropoutReproducible() {
        NeuralNetwork firstNetwork = createNetwork(0.4, 101L);
        NeuralNetwork secondNetwork = createNetwork(0.4, 101L);
        double[] inputs = {0.3, 0.7};
        double[] expectedOutputs = {0.2};

        for (int iteration = 0; iteration < 100; iteration++) {
            firstNetwork.train(inputs, expectedOutputs);
            secondNetwork.train(inputs, expectedOutputs);
        }

        assertArrayEquals(
                firstNetwork.feedForward(inputs),
                secondNetwork.feedForward(inputs),
                TOLERANCE
        );
    }

    @Test
    void convenienceConstructorCreatesAUsableNetwork() {
        /*
         * Questo costruttore sceglie autonomamente un seed. Non confrontiamo
         * quindi il valore numerico, ma verifichiamo il suo contratto osservabile:
         * una rete valida deve poter produrre un output finito della misura attesa.
         */
        NeuralNetwork network = new NeuralNetwork(2, 3, 1, 0.2, 0.0);
        double[] output = network.feedForward(new double[]{0.2, 0.8});

        assertEquals(1, output.length);
        assertTrue(Double.isFinite(output[0]));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("invalidConfigurations")
    void everyInvalidConfigurationIsRejected(
            String description,
            NeuralNetworkConfigurationSupplier configurationSupplier
    ) {
        /*
         * Ogni riga della tabella sottostante isola un solo errore. In questo
         * modo, se un controllo viene rimosso per sbaglio, il nome del caso
         * fallito spiega subito quale regola non è più protetta.
         */
        assertThrows(IllegalArgumentException.class, configurationSupplier::create);
    }

    @Test
    void invalidInputsOutputsAndParameterSnapshotsAreRejected() {
        NeuralNetwork network = createNetwork(0.0, 7L);
        assertThrows(IllegalArgumentException.class, () -> network.feedForward(new double[]{0.5}));
        assertThrows(IllegalArgumentException.class, () -> network.feedForward(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> network.feedForward(new double[]{Double.NaN, 0.5})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> network.train(new double[]{0.2, 0.8}, new double[]{0.1, 0.2})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> network.train(new double[]{0.2, 0.8}, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> network.train(
                        new double[]{0.2, 0.8},
                        new double[]{Double.POSITIVE_INFINITY}
                )
        );
        assertThrows(IllegalArgumentException.class, () -> network.restoreParameters(null));
        assertThrows(IllegalArgumentException.class, () -> network.restoreParameters(new double[1]));
    }

    private NeuralNetwork createNetwork(double dropoutRate, long seed) {
        return new NeuralNetwork(
                new NeuralNetworkConfiguration(2, 3, 1, 0.3, dropoutRate, seed)
        );
    }

    private static Stream<Arguments> invalidConfigurations() {
        return Stream.of(
                Arguments.of(
                        "input layer vuoto",
                        supplier(0, 2, 1, 0.1, 0.0)
                ),
                Arguments.of(
                        "hidden layer vuoto",
                        supplier(2, 0, 1, 0.1, 0.0)
                ),
                Arguments.of(
                        "output layer vuoto",
                        supplier(2, 2, 0, 0.1, 0.0)
                ),
                Arguments.of(
                        "learning rate uguale a zero",
                        supplier(2, 2, 1, 0.0, 0.0)
                ),
                Arguments.of(
                        "learning rate non finito",
                        supplier(2, 2, 1, Double.NaN, 0.0)
                ),
                Arguments.of(
                        "dropout negativo",
                        supplier(2, 2, 1, 0.1, -0.1)
                ),
                Arguments.of(
                        "dropout uguale a uno",
                        supplier(2, 2, 1, 0.1, 1.0)
                ),
                Arguments.of(
                        "dropout non finito",
                        supplier(2, 2, 1, 0.1, Double.NaN)
                )
        );
    }

    private static NeuralNetworkConfigurationSupplier supplier(
            int inputSize,
            int hiddenSize,
            int outputSize,
            double learningRate,
            double dropoutRate
    ) {
        return () -> new NeuralNetworkConfiguration(
                inputSize,
                hiddenSize,
                outputSize,
                learningRate,
                dropoutRate,
                1L
        );
    }

    @FunctionalInterface
    private interface NeuralNetworkConfigurationSupplier {
        NeuralNetworkConfiguration create();
    }
}
