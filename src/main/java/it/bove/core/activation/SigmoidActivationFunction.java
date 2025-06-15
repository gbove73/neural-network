package it.bove.core.activation;

/**
 * Implementazione della funzione sigmoide.
 */
public class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public double activate(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    @Override
    public double derivative(double activated) {
        return activated * (1.0 - activated);
    }
}
