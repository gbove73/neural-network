import it.bove.core.nn.NeuralNetwork;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Classe di test per la classe NeuralNetwork.
 */
public class NeuralNetworkTest {
    private NeuralNetwork neuralNetwork;

    /**
     * Inizializza una nuova istanza di NeuralNetwork prima di ogni test.
     */
    @BeforeEach
    public void setUp() {
        neuralNetwork = new NeuralNetwork(2, 2, 1, 0.5, 0.2); // Aggiungi il tasso di dropout
    }

    /**
     * Testa il metodo feedForward della classe NeuralNetwork.
     * Verifica che l'output non sia null e che la lunghezza dell'output sia corretta.
     */
    @Test
    public void testFeedForward() {
        double[] inputs = {0.5, 0.8};
        double[] outputs = neuralNetwork.feedForward(inputs);
        assertNotNull(outputs);
        assertEquals(1, outputs.length);
    }

    /**
     * Testa il metodo train della classe NeuralNetwork.
     * Verifica che l'output non sia null e che la lunghezza dell'output sia corretta dopo l'addestramento.
     */
    @Test
    public void testTrain() {
        double[] inputs = {0.5, 0.8};
        double[] expectedOutputs = {0.3};
        neuralNetwork.train(inputs, expectedOutputs);
        double[] outputs = neuralNetwork.feedForward(inputs);
        assertNotNull(outputs);
        assertEquals(1, outputs.length);
    }

    /**
     * Testa il metodo sigmoid della classe NeuralNetwork.
     * Verifica che il risultato della funzione sigmoide sia corretto.
     */
    @Test
    public void testSigmoid() {
        double result = neuralNetwork.sigmoid(0.0);
        assertEquals(0.5, result, 0.0001);
    }

    /**
     * Testa il metodo sigmoidDerivative della classe NeuralNetwork.
     * Verifica che il risultato della derivata della funzione sigmoide sia corretto.
     */
    @Test
    public void testSigmoidDerivative() {
        double result = neuralNetwork.sigmoidDerivative(0.5);
        assertEquals(0.25, result, 0.0001);
    }

    /**
     * Testa il meccanismo di dropout della rete neurale.
     * Verifica che l'addestramento con diversi tassi di dropout non generi errori
     * e che i risultati siano coerenti con l'influenza attesa del dropout.
     */
    @Test
    public void testDropout() {
        // Creiamo due reti neurali identiche ma con tassi di dropout differenti
        NeuralNetwork networkWithoutDropout = new NeuralNetwork(2, 4, 1, 0.5, 0.0); // Senza dropout
        NeuralNetwork networkWithHighDropout = new NeuralNetwork(2, 4, 1, 0.5, 0.8); // Con dropout elevato

        // Dati di addestramento semplici
        double[] inputs = {0.5, 0.8};
        double[] expectedOutputs = {0.3};

        // Addestriamo entrambe le reti con gli stessi dati per più epoche
        for (int i = 0; i < 100; i++) {
            networkWithoutDropout.train(inputs, expectedOutputs);
            networkWithHighDropout.train(inputs, expectedOutputs);
        }

        // Verifichiamo che entrambe le reti producano output validi dopo l'addestramento
        double[] outputWithoutDropout = networkWithoutDropout.feedForward(inputs);
        double[] outputWithHighDropout = networkWithHighDropout.feedForward(inputs);

        // Entrambi gli output non dovrebbero essere null
        assertNotNull(outputWithoutDropout);
        assertNotNull(outputWithHighDropout);

        // Entrambi gli output dovrebbero avere la lunghezza corretta
        assertEquals(1, outputWithoutDropout.length);
        assertEquals(1, outputWithHighDropout.length);

        // Entrambi gli output dovrebbero essere compresi tra 0 e 1 (dato che usiamo la sigmoide)
        assertTrue(outputWithoutDropout[0] >= 0 && outputWithoutDropout[0] <= 1);
        assertTrue(outputWithHighDropout[0] >= 0 && outputWithHighDropout[0] <= 1);

        // Tassi di dropout diversi dovrebbero portare a modelli con comportamenti differenti
        // Nota: Questo è un test probabilistico, non deterministico, a causa della natura del dropout
        // Il test potrebbe occasionalmente fallire
        assertNotEquals(outputWithoutDropout[0], outputWithHighDropout[0], 0.00001);
    }
}