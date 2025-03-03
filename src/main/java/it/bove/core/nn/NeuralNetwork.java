package it.bove.core.nn;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Una semplice rete neurale con un singolo strato nascosto.
 * Uno strato nascosto in una rete neurale è uno strato di neuroni che si trova tra lo strato di input e lo strato di output.
 * Non è direttamente visibile né agli input né agli output della rete.
 * Il suo scopo principale è quello di elaborare e trasformare i dati ricevuti dallo strato di input,
 * applicando una serie di pesi e funzioni di attivazione, per poi passare i risultati allo strato di output.
 * Gli strati nascosti permettono alla rete neurale di apprendere e rappresentare relazioni complesse nei dati.
 */
public class NeuralNetwork {
    private static final Logger log = LoggerFactory.getLogger(NeuralNetwork.class);

    // Matrice dei pesi tra lo strato di input e lo strato nascosto
    private final double[][] weightsInputHidden;
    // Matrice dei pesi tra lo strato nascosto e lo strato di output
    private final double[][] weightsHiddenOutput;
    // Array che rappresenta i neuroni dello strato nascosto
    private final double[] hiddenLayer;
    // Array che rappresenta i neuroni dello strato di output
    private final double[] outputLayer;
    // Tasso di apprendimento per l'aggiornamento dei pesi
    private final double learningRate;
    // Tasso di dropout
    private final double dropoutRate;
    // Random per il dropout
    private final Random rand;

    /**
     * Costruttore per la classe NeuralNetwork.
     *
     * @param inputSize    Numero di neuroni nello strato di input.
     * @param hiddenSize   Numero di neuroni nello strato nascosto.
     * @param outputSize   Numero di neuroni nello strato di output.
     * @param learningRate Tasso di apprendimento per l'aggiornamento dei pesi.
     * @param dropoutRate  Tasso di dropout per disattivare casualmente i neuroni durante l'addestramento.
     */
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate, double dropoutRate) {
        log.debug("Creazione di una nuova rete neurale con {} neuroni di input, {} neuroni nascosti, {} neuroni di output, tasso di apprendimento {} e tasso di dropout {}", inputSize, hiddenSize, outputSize, learningRate, dropoutRate);

        // Inizializza la matrice dei pesi tra input e nascosto
        this.weightsInputHidden = new double[inputSize][hiddenSize];
        // Inizializza la matrice dei pesi tra nascosto e output
        this.weightsHiddenOutput = new double[hiddenSize][outputSize];
        // Inizializza l'array dei neuroni nascosti
        this.hiddenLayer = new double[hiddenSize];
        // Inizializza l'array dei neuroni di output
        this.outputLayer = new double[outputSize];
        // Imposta il tasso di apprendimento
        this.learningRate = learningRate;
        // Imposta il tasso di dropout
        this.dropoutRate = dropoutRate;
        // Inizializza l'oggetto Random
        this.rand = new Random();

        // Chiama il metodo per inizializzare i pesi con valori casuali
        initializeWeights();
    }

    /**
     * Inizializza i pesi della rete neurale con valori casuali compresi tra -0.5 e 0.5.
     * I pesi determinano l'importanza di un neurone rispetto a un altro.
     */
    private void initializeWeights() {
        log.debug("Inizializzazione dei pesi della rete neurale");
        Random rand = new Random(); // Crea un oggetto Random per generare numeri casuali

        // Inizializza i pesi tra lo strato di input e lo strato nascosto
        initializeLayerWeights(weightsInputHidden, rand);
        // Inizializza i pesi tra lo strato nascosto e lo strato di output
        initializeLayerWeights(weightsHiddenOutput, rand);
    }

    /**
     * Inizializza i pesi di un determinato strato con valori casuali.
     *
     * @param weights Matrice dei pesi da inizializzare.
     * @param rand    Oggetto Random per generare numeri casuali.
     */
    private void initializeLayerWeights(double[][] weights, Random rand) {
        for (int i = 0; i < weights.length; i++) { // Ciclo per ogni neurone di input
            for (int j = 0; j < weights[i].length; j++) { // Ciclo per ogni neurone nascosto
                weights[i][j] = rand.nextDouble() - 0.5; // Assegna un valore casuale al peso
                log.debug("Impostato peso tra neurone {} e neurone {} a {}", i, j, weights[i][j]);
            }
        }
    }

    /**
     * Calcola la funzione di attivazione sigmoide.
     * La funzione sigmoide è una funzione matematica che mappa qualsiasi valore in un intervallo tra 0 e 1.
     *
     * @param x Valore di input.
     * @return Valore di output dopo l'applicazione della funzione sigmoide.
     */
    public double sigmoid(double x) {
        log.debug("Calcolo della funzione sigmoide per il valore {}", x);
        return 1.0 / (1.0 + Math.exp(-x)); // Formula della funzione sigmoide
    }

    /**
     * Calcola la derivata della funzione di attivazione sigmoide.
     * La derivata della funzione sigmoide è utilizzata durante l'addestramento della rete neurale.
     *
     * @param x Valore di input.
     * @return Valore della derivata della funzione sigmoide.
     */
    public double sigmoidDerivative(double x) {
        log.debug("Calcolo della derivata della funzione sigmoide per il valore {}", x);
        return x * (1.0 - x); // Formula della derivata della funzione sigmoide
    }

    /**
     * Esegue il feedforward della rete neurale.
     * Il feedforward è il processo di passare gli input attraverso la rete per ottenere gli output.
     *
     * @param inputs Array di input per la rete neurale.
     * @return Array di output della rete neurale.
     */
    public double[] feedForward(double[] inputs) {
        log.debug("Esecuzione del feedforward con input {}", inputs);

        // Calcola i valori dei neuroni nello strato nascosto
        calculateLayerOutputs(inputs, hiddenLayer, weightsInputHidden);
        // Calcola i valori dei neuroni nello strato di output
        calculateLayerOutputs(hiddenLayer, outputLayer, weightsHiddenOutput);

        log.debug("Risultato del feedforward: {}", outputLayer);
        return outputLayer; // Restituisce l'array dei valori dei neuroni di output
    }

    /**
     * Calcola i valori dei neuroni in un determinato strato.
     *
     * @param inputs  Array di input per il calcolo.
     * @param outputs Array di output per il calcolo.
     * @param weights Matrice dei pesi tra gli input e gli output.
     */
    private void calculateLayerOutputs(double[] inputs, double[] outputs, double[][] weights) {
        for (int i = 0; i < outputs.length; i++) { // Ciclo per ogni neurone di output
            outputs[i] = 0; // Inizializza il valore del neurone di output a 0
            for (int j = 0; j < inputs.length; j++) { // Ciclo per ogni neurone di input
                outputs[i] += inputs[j] * weights[j][i]; // Somma il prodotto dell'input e del peso al valore del neurone di output
            }
            outputs[i] = sigmoid(outputs[i]); // Applica la funzione sigmoide al valore del neurone di output
            log.debug("Valore del neurone {} dopo l'applicazione della funzione sigmoide: {}", i, outputs[i]);
        }
    }

    /**
     * Applica il dropout ai neuroni di un determinato strato.
     * Il dropout è una tecnica di regolarizzazione utilizzata per prevenire l'overfitting
     * nelle reti neurali. Durante l'addestramento, disattiva casualmente una frazione
     * dei neuroni, riducendo la possibilità che la rete neurale si adatti troppo ai dati
     * di addestramento. I neuroni disattivati vengono impostati a 0.
     *
     * @param layer Array dei neuroni del livello. Questo array rappresenta i valori
     *              dei neuroni in uno specifico strato della rete neurale.
     */
    private void applyDropout(double[] layer) {
        // Itera su ogni neurone del livello
        for (int i = 0; i < layer.length; i++) {
            // Verifica se il neurone deve essere disattivato in base al tasso di dropout
            if (isNeuronDropped()) {
                // Imposta il valore del neurone a 0, disattivandolo
                layer[i] = 0;
                log.debug("Neurone {} disattivato dal dropout", i);
            }
        }
    }

    /**
     * Determina se un neurone deve essere disattivato in base al tasso di dropout.
     *
     * @return true se il neurone deve essere disattivato, false altrimenti.
     */
    private boolean isNeuronDropped() {
        // Genera un numero casuale tra 0.0 e 1.0 e verifica se è inferiore al tasso di dropout.
        // Se il numero casuale è inferiore al tasso di dropout, il neurone viene disattivato.
        return rand.nextDouble() < dropoutRate;
    }

    /**
     * Allena la rete neurale utilizzando l'algoritmo di retropropagazione.
     * La retropropagazione è un algoritmo per l'addestramento delle reti neurali che minimizza l'errore.
     *
     * @param inputs          Array di input per la rete neurale.
     * @param expectedOutputs Array di output attesi per la rete neurale.
     */
    public void train(double[] inputs, double[] expectedOutputs) {
        log.debug("Inizio dell'addestramento con input {} e output attesi {}", inputs, expectedOutputs);

        double[] outputs = feedForward(inputs); // Esegue il feedforward per ottenere gli output attuali
        double[] outputErrors = calculateErrors(expectedOutputs, outputs); // Calcola gli errori dei neuroni di output
        double[] hiddenErrors = calculateHiddenErrors(outputErrors); // Calcola gli errori dei neuroni nascosti

        // Applica il dropout allo strato nascosto
        applyDropout(hiddenLayer);

        // Aggiorna i pesi tra lo strato nascosto e lo strato di output
        updateWeights(weightsHiddenOutput, hiddenLayer, outputErrors, outputLayer);
        // Aggiorna i pesi tra lo strato di input e lo strato nascosto
        updateWeights(weightsInputHidden, inputs, hiddenErrors, hiddenLayer);

        log.debug("Fine dell'addestramento");
    }

    /**
     * Calcola gli errori dei neuroni di output.
     * L'errore è la differenza tra l'output atteso e l'output attuale.
     *
     * @param expectedOutputs Array di output attesi.
     * @param actualOutputs   Array di output attuali.
     * @return Array degli errori dei neuroni di output.
     */
    private double[] calculateErrors(double[] expectedOutputs, double[] actualOutputs) {
        double[] errors = new double[actualOutputs.length];
        for (int i = 0; i < errors.length; i++) { // Ciclo per ogni neurone di output
            errors[i] = expectedOutputs[i] - actualOutputs[i]; // Calcola l'errore come differenza tra output atteso e output attuale
            log.debug("Errore per il neurone {}: atteso {}, ottenuto {}, errore {}", i, expectedOutputs[i], actualOutputs[i], errors[i]);
        }
        return errors;
    }

    /**
     * Calcola gli errori dei neuroni nascosti.
     * L'errore dei neuroni nascosti è calcolato in base agli errori dei neuroni di output.
     *
     * @param outputErrors Array degli errori dei neuroni di output.
     * @return Array degli errori dei neuroni nascosti.
     */
    private double[] calculateHiddenErrors(double[] outputErrors) {
        double[] hiddenErrors = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenErrors.length; i++) { // Ciclo per ogni neurone nascosto
            hiddenErrors[i] = 0; // Inizializza l'errore del neurone nascosto a 0
            for (int j = 0; j < outputErrors.length; j++) { // Ciclo per ogni neurone di output
                hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j]; // Somma il prodotto dell'errore di output e del peso all'errore del neurone nascosto
            }
            log.debug("Errore del neurone nascosto {}: {}", i, hiddenErrors[i]);
        }
        return hiddenErrors;
    }

    /**
     * Aggiorna i pesi della rete neurale.
     * I pesi vengono aggiornati in base agli errori e al tasso di apprendimento.
     *
     * @param weights      Matrice dei pesi da aggiornare.
     * @param layerInputs  Array degli input per il livello.
     * @param layerErrors  Array degli errori per il livello.
     * @param layerOutputs Array degli output per il livello.
     */
    private void updateWeights(double[][] weights, double[] layerInputs, double[] layerErrors, double[] layerOutputs) {
        for (int i = 0; i < weights.length; i++) { // Ciclo per ogni neurone di input
            for (int j = 0; j < weights[i].length; j++) { // Ciclo per ogni neurone di output
                // Aggiorna il peso tra neurone di input e neurone di output
                weights[i][j] += learningRate * layerErrors[j] * sigmoidDerivative(layerOutputs[j]) * layerInputs[i];
                log.debug("Aggiornamento del peso tra il neurone {} e il neurone {}: nuovo peso {}", i, j, weights[i][j]);
            }
        }
    }


}