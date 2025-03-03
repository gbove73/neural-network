import it.bove.RealEstateNeuralNetwork;
// Import delle classi necessarie per i test con JUnit 5
import org.junit.jupiter.api.BeforeEach; // Per annotare metodi che vengono eseguiti prima di ogni test
import org.junit.jupiter.api.DisplayName; // Per assegnare nomi descrittivi ai test
import org.junit.jupiter.api.Test; // Per annotare i metodi di test
import org.junit.jupiter.params.ParameterizedTest; // Per test che accettano parametri multipli
import org.junit.jupiter.params.provider.CsvSource; // Per fornire dati di test in formato CSV

// Import statico dei metodi di asserzione di JUnit per verificare condizioni nei test
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test della classe RealEstateNeuralNetwork che verifica il funzionamento
 * del sistema di valutazione immobiliare basato su reti neurali.
 * <p>
 * CASO D'USO:
 * Questo sistema utilizza una rete neurale artificiale per stimare i prezzi degli immobili
 * in base a cinque caratteristiche fondamentali:
 * 1. Dimensione (metri quadri)
 * 2. Numero di stanze
 * 3. Numero di bagni
 * 4. Piano dell'immobile
 * 5. Qualità della zona (valutata da 1 a 10)
 * <p>
 * La rete neurale impara dai dati del mercato immobiliare esistenti e crea
 * un modello che può prevedere i prezzi di nuovi immobili con caratteristiche simili.
 * Questo approccio automatizzato evita valutazioni soggettive e può rapidamente
 * analizzare grandi quantità di dati di mercato per fornire stime accurate.
 */
public class RealEstateNeuralNetworkTest {

    // Dichiariamo l'istanza del sistema di valutazione immobiliare da testare
    // Questa variabile verrà riutilizzata in tutti i test
    private RealEstateNeuralNetwork realEstateNetwork;

    // Questo metodo viene eseguito prima di ogni test, garantendo che tutti i test
    // inizino con uno stato pulito e identico del sistema di valutazione
    @BeforeEach
    public void setUp() {
        // Inizializziamo una nuova istanza del sistema con la configurazione predefinita
        // Questo crea una rete neurale con 5 input, 8 neuroni nascosti e 1 output (il prezzo)
        realEstateNetwork = new RealEstateNeuralNetwork();
    }

    // Primo test: verifica che il sistema si inizializzi correttamente
    // Il nome descrittivo chiarisce l'intento del test per chi legge i report
    @Test
    @DisplayName("Test di inizializzazione del sistema di valutazione immobiliare")
    public void testInitRealEstateNetwork() {
        // Verifichiamo che l'oggetto sia stato creato correttamente e non sia null
        // Questo garantisce che il costruttore funzioni senza errori
        assertNotNull(realEstateNetwork);

        // Testiamo una stima di prezzo su un immobile con caratteristiche medie
        // Questo verifica che la rete neurale non inizializzata possa comunque fare previsioni
        double price = realEstateNetwork.estimatePrice(100.0, 3, 1, 2, 5);

        // Verifichiamo che la stima sia un numero valido (non NaN o infinito)
        // Una rete non addestrata darà previsioni imprecise, ma dovrebbero essere numeri reali
        assertTrue(!Double.isNaN(price) && !Double.isInfinite(price), "La stima iniziale deve essere un numero valido");
    }

    // Test per verificare che la rete neurale sia in grado di apprendere dai dati
    // Questo è cruciale: dimostra che la rete migliora le sue previsioni con l'addestramento
    @Test
    @DisplayName("Test di convergenza dell'errore nell'addestramento immobiliare")
    public void testRealEstateTrainingConvergence() {
        // Dataset di addestramento con caratteristiche degli immobili
        // Ogni riga contiene: metri quadri, stanze, bagni, piano, indice di zona
        double[][] trainingFeatures = {{80.0, 3.0, 1.0, 2.0, 7.0},   // 220.000€ - appartamento medio in buona zona
                {150.0, 4.0, 2.0, 3.0, 8.0},  // 380.000€ - appartamento grande in ottima zona
                {50.0, 2.0, 1.0, 1.0, 5.0},   // 150.000€ - appartamento piccolo in zona media
                {200.0, 5.0, 3.0, 4.0, 9.0},  // 650.000€ - appartamento lussuoso in zona prestigiosa
                {90.0, 3.0, 1.0, 3.0, 6.0}    // 260.000€ - appartamento medio in buona zona, piano alto
        };

        // Prezzi corrispondenti agli immobili del dataset, in euro
        // Questi sono i valori che la rete imparerà ad associare alle caratteristiche
        double[] trainingPrices = {220000.0, 380000.0, 150000.0, 650000.0, 260000.0};

        // Addestriamo la rete con 5000 epoche (cicli completi del dataset)
        // Ogni epoca aiuta la rete a perfezionare i suoi pesi sinaptici
        realEstateNetwork.train(trainingFeatures, trainingPrices, 5000);

        // Valutiamo l'accuratezza del modello addestrato sugli stessi dati
        // Questo calcola l'Errore Medio Percentuale Assoluto (MAPE)
        double mape = realEstateNetwork.evaluateModel(trainingFeatures, trainingPrices);

        // Verifichiamo che l'errore sia accettabile (inferiore al 30%)
        // Un MAPE inferiore al 30% è considerato ragionevole per questo tipo di previsione
        assertTrue(mape < 30.0, "L'errore medio percentuale dovrebbe essere inferiore al 30%");
    }

    // Test parametrizzato che verifica la precisione su proprietà simili a quelle usate per l'addestramento
    // Questo test viene ripetuto automaticamente per ogni riga di dati fornita
    @ParameterizedTest
    @DisplayName("Test di stima del prezzo per proprietà simili a quelle di addestramento")
    @CsvSource({"85, 3, 1, 2, 7, 220000", // Caso 1: simile al primo immobile del dataset
            "130, 4, 2, 3, 8, 380000", // Caso 2: simile al secondo immobile
            "60, 2, 1, 1, 5, 150000"   // Caso 3: simile al terzo immobile
    })
    public void testPricePredictionForSimilarProperties(int mq, int rooms, int bathrooms, int floor, int zone, double expectedPrice) {

        // Dataset di addestramento identico al test precedente per consistenza
        double[][] trainingFeatures = {{80.0, 3.0, 1.0, 2.0, 7.0},   // 220.000€
                {150.0, 4.0, 2.0, 3.0, 8.0},  // 380.000€
                {50.0, 2.0, 1.0, 1.0, 5.0},   // 150.000€
                {200.0, 5.0, 3.0, 4.0, 9.0},  // 650.000€
                {90.0, 3.0, 1.0, 3.0, 6.0}    // 260.000€
        };

        // Prezzi corrispondenti agli immobili sopra
        double[] trainingPrices = {220000.0, 380000.0, 150000.0, 650000.0, 260000.0};

        // Addestriamo la rete con 10000 epoche, più del test precedente
        // Questo permette una maggiore precisione per le stime specifiche
        realEstateNetwork.train(trainingFeatures, trainingPrices, 10000);

        // Stimiamo il prezzo dell'immobile con le caratteristiche fornite dal test parametrizzato
        // I parametri cambiano ad ogni esecuzione in base ai dati CSV
        double predictedPrice = realEstateNetwork.estimatePrice(mq, rooms, bathrooms, floor, zone);

        // Definiamo una tolleranza del 20% sul prezzo atteso
        // Questo è un margine ragionevole nel mercato immobiliare
        double tolerance = 0.2 * expectedPrice;

        // Verifichiamo che la stima rientri nella tolleranza accettabile
        // Una deviazione fino al 20% è considerata accettabile vista la complessità del mercato
        assertTrue(Math.abs(predictedPrice - expectedPrice) < tolerance, String.format("La stima di €%.2f dovrebbe essere entro il 20%% di €%.2f", predictedPrice, expectedPrice));
    }

    // Test che verifica il comportamento con valori estremi delle caratteristiche
    // Questo test è importante per garantire la robustezza del sistema
    @Test
    @DisplayName("Test di robustezza con proprietà di valori estremi")
    public void testRobustnessWithExtremeProperties() {
        // Testiamo la proprietà più piccola possibile secondo i parametri di normalizzazione
        // 30mq, 1 stanza, 1 bagno, piano terra, zona peggiore
        double estimateSmall = realEstateNetwork.estimatePrice(30.0, 1, 1, 0, 1);

        // Testiamo la proprietà più grande possibile secondo i parametri di normalizzazione
        // 250mq, 5 stanze, 3 bagni, decimo piano, zona migliore
        double estimateLarge = realEstateNetwork.estimatePrice(250.0, 5, 3, 10, 10);

        // Verifichiamo che la stima per la proprietà piccola sia un valore numerico valido
        // La rete dovrebbe dare risultati sensati anche per immobili ai limiti del range
        assertTrue(!Double.isNaN(estimateSmall) && !Double.isInfinite(estimateSmall), "La stima per la proprietà piccola deve essere un numero valido");

        // Verifichiamo che la stima per la proprietà grande sia un valore numerico valido
        // Anche per gli immobili più lussuosi, il sistema dovrebbe dare risposte ragionevoli
        assertTrue(!Double.isNaN(estimateLarge) && !Double.isInfinite(estimateLarge), "La stima per la proprietà grande deve essere un numero valido");
    }

    // Test che verifica che la rete comprenda l'importanza della zona sul prezzo
    // Questo test è fondamentale perché la posizione è uno dei fattori più influenti sul valore
    @Test
    @DisplayName("Test dell'effetto zona sulla stima del prezzo")
    public void testZoneEffectOnPrice() {
        // Addestriamo con due immobili identici ma in zone diverse
        // Questo isola l'effetto della zona, mantenendo tutte le altre caratteristiche identiche
        double[][] trainingFeatures = {{100.0, 3.0, 1.0, 2.0, 3.0},  // zona economica
                {100.0, 3.0, 1.0, 2.0, 8.0}   // zona prestigiosa
        };

        // I prezzi mostrano una grande differenza dovuta solo alla zona
        // Questa differenza di prezzo significativa dovrebbe essere appresa dalla rete
        double[] trainingPrices = {180000.0,  // prezzo basso per zona economica
                350000.0   // prezzo alto per zona prestigiosa (quasi il doppio)
        };

        // Addestriamo intensivamente su questo piccolo dataset di soli due esempi
        // Questo permette alla rete di focalizzarsi sulla relazione zona-prezzo
        realEstateNetwork.train(trainingFeatures, trainingPrices, 10000);

        // Testiamo su due zone intermedie per verificare l'interpolazione
        // Zona 4 (leggermente migliore della zona economica nel training)
        double priceZone4 = realEstateNetwork.estimatePrice(100.0, 3, 1, 2, 4);
        // Zona 9 (leggermente migliore della zona prestigiosa nel training)
        double priceZone9 = realEstateNetwork.estimatePrice(100.0, 3, 1, 2, 9);

        // Verifichiamo che la zona migliore abbia un prezzo più alto
        // Questo conferma che la rete ha imparato la correlazione tra qualità della zona e prezzo
        assertTrue(priceZone9 > priceZone4, "Il prezzo in zona prestigiosa dovrebbe essere più alto");
    }
}