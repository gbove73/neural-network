package it.bove.application;

/**
 * Configurazione dell'addestramento SGD.
 *
 * <p>Un'epoca è un passaggio completo su tutti gli esempi. Lo shuffle cambia
 * l'ordine degli esempi prima di ogni passaggio, mentre il seed rende questo
 * ordine casuale ripetibile. {@code metricsInterval} evita di calcolare e
 * conservare la loss dopo ogni epoca quando il training è molto lungo.</p>
 *
 * @param epochs numero di passaggi completi sul training set
 * @param shuffleEachEpoch {@code true} per cambiare l'ordine degli esempi
 * @param shuffleSeed punto di partenza della sequenza casuale dello shuffle
 * @param metricsInterval numero di epoche tra due misurazioni della loss
 */
public record TrainingConfiguration(
        int epochs,
        boolean shuffleEachEpoch,
        long shuffleSeed,
        int metricsInterval
) {

    /**
     * Verifica che durata e frequenza delle metriche siano utilizzabili.
     */
    public TrainingConfiguration {
        // Zero epoche significherebbe chiedere un training che non apprende nulla.
        if (epochs <= 0) {
            throw new IllegalArgumentException("Il numero di epoche deve essere positivo");
        }
        // Un intervallo nullo renderebbe impossibile decidere quando registrare.
        if (metricsInterval <= 0) {
            throw new IllegalArgumentException("L'intervallo delle metriche deve essere positivo");
        }
    }

    /**
     * Configurazione predefinita con shuffle deterministico.
     */
    public static TrainingConfiguration defaults(int epochs) {
        return new TrainingConfiguration(epochs, true, 42L, Math.min(1_000, epochs));
    }
}
