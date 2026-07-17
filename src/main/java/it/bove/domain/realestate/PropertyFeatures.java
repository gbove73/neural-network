package it.bove.domain.realestate;

/**
 * Caratteristiche di un immobile espresse con nomi di dominio espliciti.
 *
 * <p>La rete usa un semplice array e non sa che il primo numero rappresenta
 * metri quadrati e il secondo stanze. Questo record dà un nome a ogni posizione,
 * impedisce di scambiarle accidentalmente e verifica che descrivano un immobile
 * plausibile prima di iniziare qualsiasi calcolo.</p>
 */
public record PropertyFeatures(
        double squareMeters,
        int rooms,
        int bathrooms,
        int floor,
        int zoneRating
) {

    public static final int FEATURE_COUNT = 5;

    /**
     * Rifiuta valori non rappresentabili o privi di significato nel dominio.
     */
    public PropertyFeatures {
        if (!Double.isFinite(squareMeters) || squareMeters <= 0.0) {
            throw new IllegalArgumentException("La superficie deve essere positiva e finita");
        }
        if (rooms <= 0 || bathrooms <= 0 || floor < 0) {
            throw new IllegalArgumentException("Stanze, bagni e piano non sono validi");
        }
        if (zoneRating < 1 || zoneRating > 10) {
            throw new IllegalArgumentException("La valutazione della zona deve essere compresa tra 1 e 10");
        }
    }

    /**
     * Converte il valore di dominio nel vettore atteso dalla rete.
     */
    public double[] toArray() {
        return new double[]{squareMeters, rooms, bathrooms, floor, zoneRating};
    }

    /**
     * Crea un valore di dominio da un vettore, verificando anche i campi discreti.
     */
    public static PropertyFeatures fromArray(double[] values) {
        if (values == null || values.length != FEATURE_COUNT) {
            throw new IllegalArgumentException("Un immobile deve contenere esattamente cinque caratteristiche");
        }
        return new PropertyFeatures(
                values[0],
                requireInteger(values[1], "stanze"),
                requireInteger(values[2], "bagni"),
                requireInteger(values[3], "piano"),
                requireInteger(values[4], "zona")
        );
    }

    private static int requireInteger(double value, String label) {
        if (!Double.isFinite(value) || value != Math.rint(value)) {
            throw new IllegalArgumentException("Il valore di " + label + " deve essere un intero finito");
        }
        return Math.toIntExact((long) value);
    }
}
