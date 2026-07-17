package it.bove.application;

import java.util.List;

/**
 * Risultato immutabile dell'addestramento, utile per verificare la convergenza.
 *
 * <p>Il confronto tra errore iniziale e finale risponde alla domanda più semplice:
 * “la rete ha imparato qualcosa?”. Lo storico mostra invece come è avvenuto il
 * percorso e può rivelare oscillazioni o un apprendimento troppo lento.</p>
 */
public record TrainingResult(
        double initialMeanSquaredError,
        double finalMeanSquaredError,
        List<TrainingMetric> metrics
) {

    /**
     * Protegge lo storico da modifiche esterne.
     */
    public TrainingResult {
        /*
         * List.copyOf crea una copia non modificabile. Chi riceve il risultato
         * può leggerlo, ma non può riscrivere per errore la storia del training.
         */
        metrics = List.copyOf(metrics);
    }
}
