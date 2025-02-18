import it.bove.NeuralNetwork;

     import static org.junit.jupiter.api.Assertions.assertEquals;
     import static org.junit.jupiter.api.Assertions.assertNotNull;

     import org.junit.jupiter.api.BeforeEach;
     import org.junit.jupiter.api.Test;

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
     }