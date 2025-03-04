package it.bove.core.normalization;

/**
 * Interfaccia generica per la normalizzazione dei dati.
 * Definisce le operazioni base di normalizzazione e denormalizzazione.
 *
 * @param <T> tipo del dato da normalizzare
 * @param <R> tipo del dato normalizzato
 */
public interface Normalizer<T, R> {

    /**
     * Normalizza un valore da un dominio originale a un intervallo normalizzato.
     *
     * @param value valore da normalizzare
     * @return valore normalizzato
     */
    R normalize(T value);

    /**
     * Denormalizza un valore dall'intervallo normalizzato al dominio originale.
     *
     * @param normalizedValue valore normalizzato
     * @return valore originale ricostruito
     */
    T denormalize(R normalizedValue);
}