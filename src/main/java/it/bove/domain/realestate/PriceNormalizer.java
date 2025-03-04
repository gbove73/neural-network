package it.bove.domain.realestate;

import it.bove.core.normalization.Normalizer;

/**
 * Interfaccia per la normalizzazione dei prezzi degli immobili.
 * Seguendo il principio di Interface Segregation, creiamo un'interfaccia specifica.
 */
public interface PriceNormalizer extends Normalizer<Double, Double> {
    /**
     * Normalizza un prezzo.
     *
     * @param price Prezzo da normalizzare
     * @return Prezzo normalizzato nell'intervallo [0,1]
     */
    Double normalize(Double price);

    /**
     * Denormalizza un prezzo normalizzato.
     *
     * @param normalizedPrice Prezzo normalizzato nell'intervallo [0,1]
     * @return Prezzo reale in euro
     */
    Double denormalize(Double normalizedPrice);
}