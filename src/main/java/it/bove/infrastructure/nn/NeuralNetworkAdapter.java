package it.bove.infrastructure.nn;

import it.bove.core.nn.NeuralNetwork;
import it.bove.core.nn.NeuralNetworkModel;

/**
 * Adapter per la classe NeuralNetwork esistente.
 *
 * <p>Un adapter può essere paragonato a un adattatore per prese elettriche:
 * non cambia l'apparecchio, ma gli permette di rispettare una forma diversa.
 * L'applicazione conosce soltanto l'interfaccia {@link NeuralNetworkModel};
 * questa classe traduce {@code predict} in {@code feedForward} e delega il
 * training alla rete matematica concreta.</p>
 */
public final class NeuralNetworkAdapter implements NeuralNetworkModel {
    // La rete neurale concreta da adattare
    private final NeuralNetwork neuralNetwork;

    /**
     * Costruttore che accetta una rete neurale da adattare.
     *
     * @param neuralNetwork La rete neurale da adattare
     */
    public NeuralNetworkAdapter(NeuralNetwork neuralNetwork) {
        if (neuralNetwork == null) {
            throw new IllegalArgumentException("La rete neurale è obbligatoria");
        }
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
