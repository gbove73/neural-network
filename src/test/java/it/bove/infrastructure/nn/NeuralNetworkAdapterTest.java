package it.bove.infrastructure.nn;

import it.bove.core.nn.NeuralNetwork;
import it.bove.core.nn.NeuralNetworkConfiguration;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifica l'adapter, cioè il piccolo traduttore tra l'interfaccia applicativa
 * {@code NeuralNetworkModel} e l'implementazione matematica {@code NeuralNetwork}.
 */
class NeuralNetworkAdapterTest {

    @Test
    void adapterRequiresAConcreteNetwork() {
        /*
         * Un adapter senza oggetto da adattare non potrebbe delegare alcuna
         * operazione. L'errore viene quindi segnalato subito nel costruttore,
         * non più tardi durante una previsione.
         */
        assertThrows(IllegalArgumentException.class, () -> new NeuralNetworkAdapter(null));
    }

    @Test
    void adapterDelegatesTrainingAndPrediction() {
        NeuralNetwork network = new NeuralNetwork(
                new NeuralNetworkConfiguration(1, 2, 1, 0.3, 0.0, 5L)
        );
        NeuralNetworkAdapter adapter = new NeuralNetworkAdapter(network);

        adapter.train(new double[]{0.5}, new double[]{0.8});

        assertEquals(1, adapter.predict(new double[]{0.5}).length);
    }
}
