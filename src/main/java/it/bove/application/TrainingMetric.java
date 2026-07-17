package it.bove.application;

/**
 * Loss media osservata al termine di un'epoca.
 *
 * <p>Associare la misura al numero di epoca permette di disegnare una curva di
 * apprendimento: se la MSE scende, le previsioni sul training set si stanno
 * avvicinando ai valori attesi.</p>
 */
public record TrainingMetric(int epoch, double meanSquaredError) {
}
