package it.bove.infrastructure.normalization;

import it.bove.domain.realestate.PriceNormalizer;

import java.util.Arrays;

/**
 * Normalizzazione min-max dei prezzi.
 *
 * <p>Un prezzo come 350.000 è molto più grande di un input come “3 stanze”.
 * Per la rete questa differenza di scala rende difficile l'apprendimento.
 * La classe traduce i prezzi in valori vicini a 0-1 e può eseguire il percorso
 * inverso per trasformare la previsione finale nuovamente in euro.</p>
 */
public final class DefaultPriceNormalizer implements PriceNormalizer {

    private final double priceMinimum;
    private final double priceMaximum;
    private final OutOfRangePolicy outOfRangePolicy;

    public DefaultPriceNormalizer(double priceMinimum, double priceMaximum) {
        this(priceMinimum, priceMaximum, OutOfRangePolicy.ALLOW);
    }

    /**
     * Costruisce un normalizzatore con una politica esplicita per i valori fuori range.
     */
    public DefaultPriceNormalizer(
            double priceMinimum,
            double priceMaximum,
            OutOfRangePolicy outOfRangePolicy
    ) {
        validatePrice(priceMinimum);
        validatePrice(priceMaximum);
        if (priceMinimum >= priceMaximum) {
            throw new IllegalArgumentException("Il prezzo minimo deve essere minore del massimo");
        }
        if (outOfRangePolicy == null) {
            throw new IllegalArgumentException("La strategia fuori range è obbligatoria");
        }
        this.priceMinimum = priceMinimum;
        this.priceMaximum = priceMaximum;
        this.outOfRangePolicy = outOfRangePolicy;
    }

    /**
     * Apprende minimo e massimo soltanto dai prezzi di training.
     */
    public static DefaultPriceNormalizer fit(
            double[] trainingPrices,
            OutOfRangePolicy outOfRangePolicy
    ) {
        /*
         * Il minimo e il massimo sono gli unici due parametri da apprendere.
         * Devono provenire dal training set: includere prezzi del test set
         * anticiperebbe alla rete informazioni che dovrebbe incontrare soltanto
         * durante la valutazione finale.
         */
        if (trainingPrices == null || trainingPrices.length < 2) {
            throw new IllegalArgumentException("Servono almeno due prezzi di training");
        }
        Arrays.stream(trainingPrices).forEach(DefaultPriceNormalizer::validatePrice);
        double minimum = Arrays.stream(trainingPrices).min().orElseThrow();
        double maximum = Arrays.stream(trainingPrices).max().orElseThrow();
        return new DefaultPriceNormalizer(minimum, maximum, outOfRangePolicy);
    }

    @Override
    public Double normalize(Double price) {
        if (price == null) {
            throw new IllegalArgumentException("Il prezzo non può essere null");
        }
        validatePrice(price);
        double adjustedPrice = switch (outOfRangePolicy) {
            case ALLOW -> price;
            case CLAMP -> Math.clamp(price, priceMinimum, priceMaximum);
            case REJECT -> {
                if (price < priceMinimum || price > priceMaximum) {
                    throw new IllegalArgumentException("Prezzo fuori dal range configurato");
                }
                yield price;
            }
        };
        return (adjustedPrice - priceMinimum) / (priceMaximum - priceMinimum);
    }

    @Override
    public Double denormalize(Double normalizedPrice) {
        if (normalizedPrice == null || !Double.isFinite(normalizedPrice)) {
            throw new IllegalArgumentException("Il prezzo normalizzato deve essere finito");
        }
        return normalizedPrice * (priceMaximum - priceMinimum) + priceMinimum;
    }

    private static void validatePrice(double price) {
        if (!Double.isFinite(price) || price <= 0.0) {
            throw new IllegalArgumentException("Il prezzo deve essere positivo e finito");
        }
    }
}
