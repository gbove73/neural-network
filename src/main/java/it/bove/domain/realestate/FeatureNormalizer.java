package it.bove.domain.realestate;

import it.bove.core.normalization.Normalizer;

/**
 * Interfaccia per la normalizzazione delle caratteristiche degli immobili.
 * Seguendo il principio di Single Responsibility, separiamo la logica di normalizzazione.
 */
public interface FeatureNormalizer extends Normalizer<double[], double[]> {
    /**
     * Normalizza le caratteristiche di un immobile.
     *
     * @param features Array delle caratteristiche da normalizzare
     * @return Array delle caratteristiche normalizzate
     */
    double[] normalize(double[] features);
}