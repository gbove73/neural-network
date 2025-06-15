package it.bove.core.activation;

/**
 * Funzione di attivazione per i neuroni della rete neurale.
 */
public interface ActivationFunction {
    /**
     * Applica la funzione di attivazione.
     *
     * @param x valore di input
     * @return valore dopo l'attivazione
     */
    double activate(double x);

    /**
     * Calcola la derivata della funzione di attivazione.
     * L'argomento è il valore già attivato per semplicità.
     *
     * @param activated output della funzione di attivazione
     * @return derivata valutata sull'output
     */
    double derivative(double activated);
}
