package it.bove.core.nn;

/**
 * Parametri immutabili necessari per costruire una rete neurale riproducibile.
 *
 * <p>Un record Java è un contenitore di valori che non cambiano dopo la
 * costruzione. Questa caratteristica è utile per un esperimento: la configurazione
 * letta nei log o nei test resta esattamente quella con cui la rete è nata.</p>
 *
 * @param inputSize numero di valori ricevuti da ogni esempio
 * @param hiddenSize numero di neuroni che elaborano internamente gli input
 * @param outputSize numero di valori prodotti dalla rete
 * @param learningRate ampiezza di ogni correzione dei parametri
 * @param dropoutRate frazione media di neuroni nascosti spenta durante il training
 * @param seed valore che rende ripetibile la sequenza casuale
 */
public record NeuralNetworkConfiguration(
        int inputSize,
        int hiddenSize,
        int outputSize,
        double learningRate,
        double dropoutRate,
        long seed
) {

    /**
     * Valida dimensioni e iperparametri prima di allocare la rete.
     */
    public NeuralNetworkConfiguration {
        if (inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0) {
            throw new IllegalArgumentException("Le dimensioni dei layer devono essere positive");
        }
        if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
            throw new IllegalArgumentException("Il learning rate deve essere positivo e finito");
        }
        if (!Double.isFinite(dropoutRate) || dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw new IllegalArgumentException("Il dropout deve appartenere all'intervallo [0, 1)");
        }
    }
}
