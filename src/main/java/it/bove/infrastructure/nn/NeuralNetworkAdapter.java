package it.bove.infrastructure.nn;

import it.bove.core.nn.NeuralNetwork;
import it.bove.core.nn.NeuralNetworkModel;

/**
 * Adapter per la classe NeuralNetwork esistente.
 * Seguendo il pattern Adapter, adattiamo l'implementazione esistente alla nostra interfaccia.
 */
public class NeuralNetworkAdapter implements NeuralNetworkModel {
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