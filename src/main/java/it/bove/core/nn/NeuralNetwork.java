package it.bove.core.nn;

import java.util.Arrays;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Percettrone multistrato con un solo strato nascosto.
 *
 * <p>Una rete neurale può essere immaginata come una catena di piccoli calcolatori,
 * chiamati neuroni. Ogni neurone riceve alcuni numeri, attribuisce a ciascuno
 * un'importanza (il peso), aggiunge una correzione personale (il bias) e trasforma
 * il risultato con una funzione di attivazione. In questa classe i numeri attraversano
 * tre livelli:</p>
 *
 * <ol>
 *   <li>lo strato di input riceve i dati così come sono;</li>
 *   <li>lo strato nascosto combina quei dati e apprende relazioni non lineari;</li>
 *   <li>lo strato di output produce la previsione finale.</li>
 * </ol>
 *
 * <p>L'implementazione lascia volutamente visibili forward pass e backpropagation.
 * Una libreria professionale nasconderebbe questi dettagli dietro operazioni su
 * matrici; qui, invece, cicli e formule restano espliciti per permettere di seguire
 * il percorso di ogni valore.</p>
 */
public final class NeuralNetwork {

    /*
     * Una matrice è una tabella di numeri. La cella [i][j] contiene il peso del
     * collegamento che parte dal neurone i e arriva al neurone j.
     */
    private final double[][] inputHiddenWeights;
    private final double[][] hiddenOutputWeights;

    /*
     * Il bias è un valore aggiunto alla somma pesata. È simile alla possibilità
     * di spostare una soglia verso destra o sinistra: senza bias, un neurone
     * sarebbe inutilmente vincolato a una soglia centrata nello zero.
     */
    private final double[] hiddenBiases;
    private final double[] outputBiases;

    /*
     * Il learning rate stabilisce quanto correggere i parametri a ogni esempio:
     * troppo piccolo rallenta l'apprendimento, troppo grande può farlo oscillare.
     */
    private final double learningRate;

    /*
     * Il dropout spegne temporaneamente alcuni neuroni durante il training.
     * In questo modo la rete non può affidarsi sempre agli stessi percorsi.
     */
    private final double dropoutRate;

    /*
     * Tutta la casualità passa da un unico generatore. Partendo dallo stesso seed
     * si ottengono gli stessi pesi e le stesse maschere di dropout: un requisito
     * fondamentale per ripetere un esperimento e capire perché ha dato un risultato.
     */
    private final RandomGenerator randomGenerator;

    /**
     * Crea una rete usando un seed non deterministico.
     */
    public NeuralNetwork(
            int inputSize,
            int hiddenSize,
            int outputSize,
            double learningRate,
            double dropoutRate
    ) {
        this(new NeuralNetworkConfiguration(
                inputSize,
                hiddenSize,
                outputSize,
                learningRate,
                dropoutRate,
                System.nanoTime()
        ));
    }

    /**
     * Crea una rete a partire da una configurazione validata.
     *
     * @param configuration configurazione della rete
     */
    public NeuralNetwork(NeuralNetworkConfiguration configuration) {
        this.learningRate = configuration.learningRate();
        this.dropoutRate = configuration.dropoutRate();
        this.randomGenerator = RandomGeneratorFactory.getDefault().create(configuration.seed());
        this.inputHiddenWeights = new double[configuration.inputSize()][configuration.hiddenSize()];
        this.hiddenOutputWeights = new double[configuration.hiddenSize()][configuration.outputSize()];
        this.hiddenBiases = new double[configuration.hiddenSize()];
        this.outputBiases = new double[configuration.outputSize()];
        initializeWeights();
    }

    /**
     * Calcola una previsione senza dropout.
     *
     * @param inputs valori di ingresso
     * @return nuovo array contenente gli output
     */
    public double[] feedForward(double[] inputs) {
        validateInput(inputs);
        /*
         * In previsione il dropout è disattivato: vogliamo usare l'intera rete
         * che è stata appresa, non spegnere collegamenti in modo casuale.
         */
        return forward(inputs, false).outputs();
    }

    /**
     * Esegue un aggiornamento SGD su un singolo esempio.
     *
     * @param inputs input normalizzati
     * @param expectedOutputs output attesi normalizzati
     */
    public void train(double[] inputs, double[] expectedOutputs) {
        validateInput(inputs);
        validateExpectedOutput(expectedOutputs);

        /*
         * L'apprendimento di un esempio avviene in tre fasi:
         * 1. la rete produce una previsione;
         * 2. la backpropagation misura la responsabilità di ogni parametro;
         * 3. pesi e bias vengono mossi nella direzione che riduce l'errore.
         */
        ForwardPass forwardPass = forward(inputs, true);
        Gradients gradients = calculateGradients(inputs, expectedOutputs, forwardPass);
        applyGradients(gradients);
    }

    /**
     * Funzione sigmoide numericamente stabile.
     */
    public double sigmoid(double value) {
        /*
         * La formula classica è 1 / (1 + e^-x). Separare valori positivi e
         * negativi evita che l'esponenziale diventi così grande da andare in
         * overflow, pur calcolando esattamente la stessa funzione.
         */
        if (value >= 0) {
            double exponential = Math.exp(-value);
            return 1.0 / (1.0 + exponential);
        }
        double exponential = Math.exp(value);
        return exponential / (1.0 + exponential);
    }

    /**
     * Derivata della sigmoide calcolata a partire dal suo output.
     */
    public double sigmoidDerivative(double sigmoidOutput) {
        return sigmoidOutput * (1.0 - sigmoidOutput);
    }

    /**
     * Calcola la loss quadratica media per un singolo esempio.
     */
    public double calculateLoss(double[] inputs, double[] expectedOutputs) {
        validateInput(inputs);
        validateExpectedOutput(expectedOutputs);
        double[] outputs = feedForward(inputs);
        double squaredError = 0.0;
        for (int index = 0; index < outputs.length; index++) {
            double error = outputs[index] - expectedOutputs[index];
            squaredError += error * error;
        }
        return squaredError / (2.0 * outputs.length);
    }

    private void initializeWeights() {
        initializeLayerWeights(inputHiddenWeights, inputHiddenWeights.length);
        initializeLayerWeights(hiddenOutputWeights, hiddenOutputWeights.length);
    }

    private void initializeLayerWeights(double[][] weights, int fanIn) {
        /*
         * Se tutti i pesi partissero da zero, i neuroni imparerebbero tutti la
         * stessa cosa. Usiamo quindi valori casuali. L'intervallo Xavier tiene
         * conto di quanti collegamenti entrano ed escono dal layer: così le somme
         * iniziali non sono né quasi nulle né tanto grandi da saturare la sigmoide.
         */
        double limit = Math.sqrt(6.0 / (fanIn + weights[0].length));
        for (double[] row : weights) {
            for (int column = 0; column < row.length; column++) {
                row[column] = randomGenerator.nextDouble(-limit, limit);
            }
        }
    }

    private ForwardPass forward(double[] inputs, boolean training) {
        /*
         * Primo passaggio: ogni neurone nascosto calcola
         *
         * sigmoide(bias + input1*peso1 + input2*peso2 + ...).
         *
         * Conserviamo sia il valore originale sia quello dopo il dropout.
         * Il valore originale serve più tardi per calcolare la derivata corretta.
         */
        double[] rawHiddenOutputs = activateLayer(inputs, inputHiddenWeights, hiddenBiases);
        double[] hiddenOutputs = Arrays.copyOf(rawHiddenOutputs, rawHiddenOutputs.length);
        double[] dropoutScales = new double[hiddenOutputs.length];
        Arrays.fill(dropoutScales, 1.0);

        if (training && dropoutRate > 0.0) {
            /*
             * "Inverted dropout": se, per esempio, conserviamo in media il 75%
             * dei neuroni, quelli rimasti vengono moltiplicati per 1 / 0,75.
             * La loro intensità media resta così uguale a quella usata in previsione,
             * quando nessun neurone viene spento.
             */
            double retainedScale = 1.0 / (1.0 - dropoutRate);
            for (int index = 0; index < hiddenOutputs.length; index++) {
                dropoutScales[index] = randomGenerator.nextDouble() < dropoutRate ? 0.0 : retainedScale;
                hiddenOutputs[index] *= dropoutScales[index];
            }
        }

        // Secondo passaggio: gli output nascosti diventano gli input del layer finale.
        double[] outputs = activateLayer(hiddenOutputs, hiddenOutputWeights, outputBiases);
        return new ForwardPass(rawHiddenOutputs, hiddenOutputs, dropoutScales, outputs);
    }

    private double[] activateLayer(double[] inputs, double[][] weights, double[] biases) {
        double[] outputs = new double[biases.length];
        for (int outputIndex = 0; outputIndex < outputs.length; outputIndex++) {
            /*
             * Partiamo dal bias del neurone. Il ciclo interno visita poi tutti
             * i valori in ingresso e aggiunge input × peso. Questa è la "somma
             * pesata" citata nella teoria delle reti neurali.
             */
            double weightedSum = biases[outputIndex];
            for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
                weightedSum += inputs[inputIndex] * weights[inputIndex][outputIndex];
            }
            outputs[outputIndex] = sigmoid(weightedSum);
        }
        return outputs;
    }

    private Gradients calculateGradients(
            double[] inputs,
            double[] expectedOutputs,
            ForwardPass forwardPass
    ) {
        /*
         * Un delta indica quanto la loss cambierebbe al variare della somma
         * ricevuta da un neurone. Per il layer di output applichiamo la regola
         * della catena:
         *
         * delta output = (previsione - valore atteso) × derivata sigmoide.
         *
         * Il primo fattore deriva dalla loss quadratica; il secondo traduce una
         * variazione dell'uscita in una variazione della somma prima della sigmoide.
         */
        double[] outputDeltas = new double[outputBiases.length];
        for (int outputIndex = 0; outputIndex < outputDeltas.length; outputIndex++) {
            double errorDerivative = forwardPass.outputs()[outputIndex] - expectedOutputs[outputIndex];
            outputDeltas[outputIndex] =
                    errorDerivative * sigmoidDerivative(forwardPass.outputs()[outputIndex]);
        }

        /*
         * Lo strato nascosto non conosce direttamente il risultato atteso.
         * Riceve quindi "all'indietro" i delta degli output, pesati per la forza
         * dei collegamenti che li uniscono. Applichiamo poi la derivata della
         * sua sigmoide e la stessa maschera di dropout usata nel forward pass.
         *
         * Usare la stessa maschera è essenziale: non avrebbe senso attribuire
         * un errore a un neurone che era stato spento quando è nata la previsione.
         */
        double[] hiddenDeltas = new double[hiddenBiases.length];
        for (int hiddenIndex = 0; hiddenIndex < hiddenDeltas.length; hiddenIndex++) {
            double propagatedDelta = 0.0;
            for (int outputIndex = 0; outputIndex < outputDeltas.length; outputIndex++) {
                propagatedDelta += outputDeltas[outputIndex] * hiddenOutputWeights[hiddenIndex][outputIndex];
            }
            hiddenDeltas[hiddenIndex] = propagatedDelta
                    * sigmoidDerivative(forwardPass.rawHiddenOutputs()[hiddenIndex])
                    * forwardPass.dropoutScales()[hiddenIndex];
        }

        /*
         * Per un collegamento, gradiente = valore in ingresso × delta in uscita.
         * L'outer product calcola questa moltiplicazione per ogni coppia possibile
         * e produce una tabella della stessa forma della matrice dei pesi.
         */
        double[][] hiddenOutputGradients =
                outerProduct(forwardPass.hiddenOutputs(), outputDeltas);
        double[][] inputHiddenGradients = outerProduct(inputs, hiddenDeltas);
        return new Gradients(
                inputHiddenGradients,
                hiddenOutputGradients,
                hiddenDeltas,
                outputDeltas
        );
    }

    private double[][] outerProduct(double[] inputs, double[] deltas) {
        double[][] gradients = new double[inputs.length][deltas.length];
        for (int inputIndex = 0; inputIndex < inputs.length; inputIndex++) {
            for (int deltaIndex = 0; deltaIndex < deltas.length; deltaIndex++) {
                gradients[inputIndex][deltaIndex] = inputs[inputIndex] * deltas[deltaIndex];
            }
        }
        return gradients;
    }

    private void applyGradients(Gradients gradients) {
        /*
         * Gradient descent:
         *
         * nuovo parametro = vecchio parametro - learning rate × gradiente.
         *
         * Sottraiamo perché il gradiente punta verso la crescita più rapida
         * della loss, mentre noi vogliamo percorrere la direzione opposta.
         */
        subtractScaled(inputHiddenWeights, gradients.inputHiddenWeights());
        subtractScaled(hiddenOutputWeights, gradients.hiddenOutputWeights());
        subtractScaled(hiddenBiases, gradients.hiddenBiases());
        subtractScaled(outputBiases, gradients.outputBiases());
    }

    private void subtractScaled(double[][] parameters, double[][] gradients) {
        for (int row = 0; row < parameters.length; row++) {
            subtractScaled(parameters[row], gradients[row]);
        }
    }

    private void subtractScaled(double[] parameters, double[] gradients) {
        for (int index = 0; index < parameters.length; index++) {
            parameters[index] -= learningRate * gradients[index];
        }
    }

    private void validateInput(double[] inputs) {
        validateFiniteArray(inputs, "Gli input");
        if (inputs.length != inputHiddenWeights.length) {
            throw new IllegalArgumentException(
                    "Attesi %d input, ricevuti %d".formatted(inputHiddenWeights.length, inputs.length)
            );
        }
    }

    private void validateExpectedOutput(double[] expectedOutputs) {
        validateFiniteArray(expectedOutputs, "Gli output attesi");
        if (expectedOutputs.length != outputBiases.length) {
            throw new IllegalArgumentException(
                    "Attesi %d output, ricevuti %d".formatted(outputBiases.length, expectedOutputs.length)
            );
        }
    }

    private void validateFiniteArray(double[] values, String label) {
        if (values == null) {
            throw new IllegalArgumentException(label + " non possono essere null");
        }
        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(label + " devono contenere solo valori finiti");
            }
        }
    }

    /*
     * I metodi seguenti sono visibili soltanto ai test dello stesso package.
     * Consentono il gradient checking, cioè il confronto tra la derivata analitica
     * della backpropagation e una derivata numerica ottenuta perturbando i pesi.
     * Non fanno parte dell'API destinata a chi usa la rete.
     */
    double[] parametersSnapshot() {
        return flatten(inputHiddenWeights, hiddenOutputWeights, hiddenBiases, outputBiases);
    }

    void restoreParameters(double[] parameters) {
        int expectedSize = parametersSnapshot().length;
        if (parameters == null || parameters.length != expectedSize) {
            throw new IllegalArgumentException("Numero di parametri non valido");
        }
        int offset = 0;
        offset = restore(inputHiddenWeights, parameters, offset);
        offset = restore(hiddenOutputWeights, parameters, offset);
        offset = restore(hiddenBiases, parameters, offset);
        restore(outputBiases, parameters, offset);
    }

    double[] gradientSnapshot(double[] inputs, double[] expectedOutputs) {
        validateInput(inputs);
        validateExpectedOutput(expectedOutputs);
        Gradients gradients = calculateGradients(inputs, expectedOutputs, forward(inputs, false));
        return flatten(
                gradients.inputHiddenWeights(),
                gradients.hiddenOutputWeights(),
                gradients.hiddenBiases(),
                gradients.outputBiases()
        );
    }

    private double[] flatten(
            double[][] firstMatrix,
            double[][] secondMatrix,
            double[] firstVector,
            double[] secondVector
    ) {
        int size = matrixSize(firstMatrix) + matrixSize(secondMatrix)
                + firstVector.length + secondVector.length;
        double[] flattened = new double[size];
        int offset = 0;
        offset = flatten(firstMatrix, flattened, offset);
        offset = flatten(secondMatrix, flattened, offset);
        offset = flatten(firstVector, flattened, offset);
        flatten(secondVector, flattened, offset);
        return flattened;
    }

    private int matrixSize(double[][] matrix) {
        return matrix.length * matrix[0].length;
    }

    private int flatten(double[][] matrix, double[] destination, int offset) {
        for (double[] row : matrix) {
            offset = flatten(row, destination, offset);
        }
        return offset;
    }

    private int flatten(double[] vector, double[] destination, int offset) {
        System.arraycopy(vector, 0, destination, offset, vector.length);
        return offset + vector.length;
    }

    private int restore(double[][] matrix, double[] source, int offset) {
        for (double[] row : matrix) {
            offset = restore(row, source, offset);
        }
        return offset;
    }

    private int restore(double[] vector, double[] source, int offset) {
        System.arraycopy(source, offset, vector, 0, vector.length);
        return offset + vector.length;
    }

    private record ForwardPass(
            double[] rawHiddenOutputs,
            double[] hiddenOutputs,
            double[] dropoutScales,
            double[] outputs
    ) {
    }

    private record Gradients(
            double[][] inputHiddenWeights,
            double[][] hiddenOutputWeights,
            double[] hiddenBiases,
            double[] outputBiases
    ) {
    }
}
