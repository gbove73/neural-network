package it.bove.core.nn;

/**
 * Interfaccia che definisce il comportamento di un modello di rete neurale.
 * Seguendo il principio di Interface Segregation, definiamo un'interfaccia minimale.
 */
public interface NeuralNetworkModel {
    /**
     * Addestra il modello con un esempio.
     *
     * @param input          Caratteristiche di input normalizzate
     * @param expectedOutput Output atteso normalizzato
     */
    void train(double[] input, double[] expectedOutput);

    /**
     * Esegue una predizione.
     *
     * @param input Caratteristiche di input normalizzate
     * @return Output previsto normalizzato
     */
    double[] predict(double[] input);
}
