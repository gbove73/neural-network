package it.bove.infrastructure.normalization;

/**
 * Strategia applicata quando un valore supera il range usato dalla normalizzazione.
 */
public enum OutOfRangePolicy {
    /**
     * Continua la formula: il risultato potrà essere minore di 0 o maggiore di 1.
     */
    ALLOW,

    /**
     * Riporta il valore al limite più vicino; per esempio 120 in un range 0-100 diventa 100.
     */
    CLAMP,

    /**
     * Interrompe l'operazione per segnalare che il dato non appartiene al dominio previsto.
     */
    REJECT
}
