package it.bove.core.activation;

/**
 * Funzione di attivazione ReLU (Rectified Linear Unit).
 */
public class ReLUActivationFunction implements ActivationFunction {
    @Override
    public double activate(double x) {
        return Math.max(0.0, x);
    }

    @Override
    public double derivative(double activated) {
        return activated > 0.0 ? 1.0 : 0.0;
    }
}
