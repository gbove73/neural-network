package it.bove.domain.realestate;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifica il piccolo oggetto che dà un nome alle cinque colonne del dataset.
 *
 * <p>Questi test sono intenzionalmente espliciti: mostrano a chi legge quali
 * valori hanno significato nel dominio immobiliare prima ancora che intervenga
 * la rete neurale.</p>
 */
class PropertyFeaturesTest {

    @Test
    void validPropertyCanTravelFromObjectToArrayAndBack() {
        PropertyFeatures original = new PropertyFeatures(95.5, 3, 2, 4, 7);

        double[] serialized = original.toArray();
        PropertyFeatures reconstructed = PropertyFeatures.fromArray(serialized);

        assertArrayEquals(new double[]{95.5, 3.0, 2.0, 4.0, 7.0}, serialized);
        assertEquals(original, reconstructed);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("invalidProperties")
    void invalidDomainValuesAreRejected(String description, PropertySupplier propertySupplier) {
        /*
         * Non basta che un numero entri in un double: deve anche descrivere un
         * immobile possibile. Ogni caso isola una regola del costruttore.
         */
        assertThrows(IllegalArgumentException.class, propertySupplier::create);
    }

    @Test
    void arraysMustHaveFiveFiniteAndDiscreteValuesWhereRequired() {
        assertThrows(IllegalArgumentException.class, () -> PropertyFeatures.fromArray(null));
        assertThrows(
                IllegalArgumentException.class,
                () -> PropertyFeatures.fromArray(new double[]{80.0})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> PropertyFeatures.fromArray(new double[]{80.0, 2.5, 1.0, 2.0, 5.0})
        );
        assertThrows(
                IllegalArgumentException.class,
                () -> PropertyFeatures.fromArray(
                        new double[]{80.0, Double.NaN, 1.0, 2.0, 5.0}
                )
        );
    }

    private static Stream<Arguments> invalidProperties() {
        return Stream.of(
                Arguments.of("superficie zero", property(0.0, 2, 1, 0, 5)),
                Arguments.of("superficie non finita", property(Double.NaN, 2, 1, 0, 5)),
                Arguments.of("nessuna stanza", property(80.0, 0, 1, 0, 5)),
                Arguments.of("nessun bagno", property(80.0, 2, 0, 0, 5)),
                Arguments.of("piano negativo", property(80.0, 2, 1, -1, 5)),
                Arguments.of("zona sotto il minimo", property(80.0, 2, 1, 0, 0)),
                Arguments.of("zona sopra il massimo", property(80.0, 2, 1, 0, 11))
        );
    }

    private static PropertySupplier property(
            double squareMeters,
            int rooms,
            int bathrooms,
            int floor,
            int zoneRating
    ) {
        return () -> new PropertyFeatures(squareMeters, rooms, bathrooms, floor, zoneRating);
    }

    @FunctionalInterface
    private interface PropertySupplier {
        PropertyFeatures create();
    }
}
