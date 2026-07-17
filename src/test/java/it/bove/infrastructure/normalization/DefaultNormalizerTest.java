package it.bove.infrastructure.normalization;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifica reversibilità, apprendimento dei range e politiche fuori intervallo.
 *
 * <p>Normalizzare significa cambiare scala, non perdere informazione. Per questo
 * i test controllano prima il viaggio di andata e ritorno. Gli altri scenari
 * spiegano cosa accade quando un dato supera i limiti o quando il range stesso
 * è impossibile, per esempio minimo e massimo uguali.</p>
 */
class DefaultNormalizerTest {

    @Test
    void featureNormalizationIsReversible() {
        /*
         * Se normalizziamo e subito dopo denormalizziamo, dobbiamo ritrovare i
         * valori di partenza. Una piccola tolleranza assorbe soltanto gli
         * inevitabili arrotondamenti dei numeri decimali nel computer.
         */
        DefaultFeatureNormalizer normalizer = new DefaultFeatureNormalizer(
                new double[]{0.0, 10.0},
                new double[]{10.0, 30.0}
        );
        double[] original = {2.5, 20.0};

        assertArrayEquals(
                original,
                normalizer.denormalize(normalizer.normalize(original)),
                1.0e-9
        );
    }

    @Test
    void fittedNormalizerUsesTrainingExtremes() {
        /*
         * Il dataset insegna che 2 e 6 sono gli estremi della prima colonna,
         * mentre 10 e 30 lo sono della seconda. Il punto centrale deve quindi
         * diventare 0,5 in entrambe le colonne.
         */
        DefaultFeatureNormalizer normalizer = DefaultFeatureNormalizer.fit(
                new double[][]{{2.0, 10.0}, {6.0, 30.0}, {4.0, 20.0}},
                OutOfRangePolicy.REJECT
        );

        assertArrayEquals(new double[]{0.5, 0.5}, normalizer.normalize(new double[]{4.0, 20.0}));
    }

    @Test
    void outOfRangePoliciesAreAppliedExplicitly() {
        /*
         * Lo stesso prezzo 250 supera il massimo 200. ALLOW produce 1,5,
         * CLAMP lo limita a 1 e REJECT segnala l'errore: i tre comportamenti
         * sono intenzionalmente diversi e scelti dal chiamante.
         */
        DefaultPriceNormalizer allowNormalizer =
                new DefaultPriceNormalizer(100.0, 200.0);
        DefaultPriceNormalizer clampNormalizer =
                new DefaultPriceNormalizer(100.0, 200.0, OutOfRangePolicy.CLAMP);
        DefaultPriceNormalizer rejectNormalizer =
                new DefaultPriceNormalizer(100.0, 200.0, OutOfRangePolicy.REJECT);

        assertEquals(1.5, allowNormalizer.normalize(250.0));
        assertEquals(1.0, clampNormalizer.normalize(250.0));
        assertEquals(0.5, rejectNormalizer.normalize(150.0));
        assertThrows(IllegalArgumentException.class, () -> rejectNormalizer.normalize(50.0));
        assertThrows(IllegalArgumentException.class, () -> rejectNormalizer.normalize(250.0));
    }

    @Test
    void priceNormalizerCanBeFittedAndReversesItsTransformation() {
        DefaultPriceNormalizer normalizer = DefaultPriceNormalizer.fit(
                new double[]{100.0, 150.0, 300.0},
                OutOfRangePolicy.ALLOW
        );

        assertEquals(0.25, normalizer.normalize(150.0));
        assertEquals(150.0, normalizer.denormalize(0.25));
    }

    @Test
    void featurePoliciesCoverAllowClampRejectAndBothBounds() {
        double[] minimums = {0.0};
        double[] maximums = {10.0};
        DefaultFeatureNormalizer allowNormalizer =
                new DefaultFeatureNormalizer(minimums, maximums, OutOfRangePolicy.ALLOW);
        DefaultFeatureNormalizer clampNormalizer =
                new DefaultFeatureNormalizer(minimums, maximums, OutOfRangePolicy.CLAMP);
        DefaultFeatureNormalizer rejectNormalizer =
                new DefaultFeatureNormalizer(minimums, maximums, OutOfRangePolicy.REJECT);

        assertArrayEquals(new double[]{1.2}, allowNormalizer.normalize(new double[]{12.0}));
        assertArrayEquals(new double[]{0.0}, clampNormalizer.normalize(new double[]{-2.0}));
        assertArrayEquals(new double[]{1.0}, clampNormalizer.normalize(new double[]{12.0}));
        assertArrayEquals(new double[]{0.5}, rejectNormalizer.normalize(new double[]{5.0}));
        assertThrows(
                IllegalArgumentException.class,
                () -> rejectNormalizer.normalize(new double[]{-1.0})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> rejectNormalizer.normalize(new double[]{11.0})
        );
    }

    @Test
    void malformedFeatureRangesAndPoliciesAreRejected() {
        /*
         * Un range utilizzabile richiede due array presenti, non vuoti, della
         * stessa dimensione, composti da valori finiti e con minimo < massimo.
         */
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(null, new double[]{1.0})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(new double[]{0.0}, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(new double[0], new double[0])
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(
                        new double[]{0.0},
                        new double[]{1.0, 2.0}
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(
                        new double[]{1.0},
                        new double[]{1.0}
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(
                        new double[]{Double.NaN},
                        new double[]{1.0}
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(
                        new double[]{0.0},
                        new double[]{Double.POSITIVE_INFINITY}
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultFeatureNormalizer(
                        new double[]{0.0},
                        new double[]{1.0},
                        null
                )
        );
    }

    @Test
    void malformedFeatureDatasetsAndVectorsAreRejected() {
        /*
         * Una matrice di training deve essere rettangolare: tutte le righe
         * descrivono lo stesso numero di caratteristiche. Null, righe vuote,
         * lunghezze diverse e NaN renderebbero il significato delle colonne incerto.
         */
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(null, OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(new double[0][], OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(new double[][]{null}, OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(
                        new double[][]{{1.0, 2.0}, null},
                        OutOfRangePolicy.ALLOW
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(
                        new double[][]{{}, {}},
                        OutOfRangePolicy.ALLOW
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(
                        new double[][]{{1.0, 2.0}, {2.0}},
                        OutOfRangePolicy.ALLOW
                )
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultFeatureNormalizer.fit(
                        new double[][]{{1.0, 2.0}, {2.0, Double.NaN}},
                        OutOfRangePolicy.ALLOW
                )
        );

        DefaultFeatureNormalizer normalizer = new DefaultFeatureNormalizer(
                new double[]{0.0},
                new double[]{1.0}
        );
        assertThrows(IllegalArgumentException.class, () -> normalizer.normalize(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> normalizer.normalize(new double[]{0.2, 0.3})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> normalizer.denormalize(new double[]{Double.NaN})
        );
    }

    @Test
    void malformedPriceRangesValuesAndPoliciesAreRejected() {
        /*
         * I prezzi sono grandezze positive. Anche infinito e NaN sono rifiutati:
         * pur essendo rappresentabili da double, non descrivono importi reali.
         */
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultPriceNormalizer(0.0, 200.0)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultPriceNormalizer(100.0, Double.NaN)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultPriceNormalizer(200.0, 100.0)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> new DefaultPriceNormalizer(100.0, 200.0, null)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultPriceNormalizer.fit(null, OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultPriceNormalizer.fit(new double[0], OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultPriceNormalizer.fit(new double[]{100.0}, OutOfRangePolicy.ALLOW)
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> DefaultPriceNormalizer.fit(
                        new double[]{100.0, 100.0},
                        OutOfRangePolicy.ALLOW
                )
        );

        DefaultPriceNormalizer normalizer = new DefaultPriceNormalizer(100.0, 200.0);
        assertThrows(IllegalArgumentException.class, () -> normalizer.normalize(null));
        assertThrows(IllegalArgumentException.class, () -> normalizer.normalize(Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> normalizer.normalize(0.0));
        assertThrows(IllegalArgumentException.class, () -> normalizer.denormalize(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> normalizer.denormalize(Double.POSITIVE_INFINITY)
        );
    }
}
