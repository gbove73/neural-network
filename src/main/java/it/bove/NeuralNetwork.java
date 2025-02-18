package it.bove;

import java.util.Random;

/**
 * La classe NeuralNetwork implementa una semplice rete neurale con un singolo strato nascosto.
 */
public class NeuralNetwork {
    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;
    private double[] hiddenLayer;
    private double[] outputLayer;
    private double learningRate;

    /**
     * Costruttore per la classe NeuralNetwork.
     *
     * @param inputSize     Numero di neuroni nello strato di input.
     * @param hiddenSize    Numero di neuroni nello strato nascosto.
     * @param outputSize    Numero di neuroni nello strato di output.
     * @param learningRate  Tasso di apprendimento per l'aggiornamento dei pesi.
     */
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate) {
        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.weightsHiddenOutput = new double[hiddenSize][outputSize];
        this.hiddenLayer = new double[hiddenSize];
        this.outputLayer = new double[outputSize];
        this.learningRate = learningRate;
        initializeWeights();
    }

    /**
     * Inizializza i pesi della rete neurale con valori casuali compresi tra -0.5 e 0.5.
     */
    private void initializeWeights() {
        Random rand = new Random();
        for (int i = 0; i < weightsInputHidden.length; i++) {
            for (int j = 0; j < weightsInputHidden[i].length; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() - 0.5;
            }
        }
        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            for (int j = 0; j < weightsHiddenOutput[i].length; j++) {
                weightsHiddenOutput[i][j] = rand.nextDouble() - 0.5;
            }
        }
    }

    /**
     * Calcola la funzione di attivazione sigmoide.
     *
     * @param x Valore di input.
     * @return Valore di output dopo l'applicazione della funzione sigmoide.
     */
    public double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Calcola la derivata della funzione di attivazione sigmoide.
     *
     * @param x Valore di input.
     * @return Valore della derivata della funzione sigmoide.
     */
    public double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    /**
     * Esegue il feedforward della rete neurale.
     *
     * @param inputs Array di input per la rete neurale.
     * @return Array di output della rete neurale.
     */
    public double[] feedForward(double[] inputs) {
        for (int i = 0; i < hiddenLayer.length; i++) {
            hiddenLayer[i] = 0;
            for (int j = 0; j < inputs.length; j++) {
                hiddenLayer[i] += inputs[j] * weightsInputHidden[j][i];
            }
            hiddenLayer[i] = sigmoid(hiddenLayer[i]);
        }

        for (int i = 0; i < outputLayer.length; i++) {
            outputLayer[i] = 0;
            for (int j = 0; j < hiddenLayer.length; j++) {
                outputLayer[i] += hiddenLayer[j] * weightsHiddenOutput[j][i];
            }
            outputLayer[i] = sigmoid(outputLayer[i]);
        }

        return outputLayer;
    }

    /**
     * Allena la rete neurale utilizzando l'algoritmo di retropropagazione.
     *
     * @param inputs          Array di input per la rete neurale.
     * @param expectedOutputs Array di output attesi per la rete neurale.
     */
    public void train(double[] inputs, double[] expectedOutputs) {
        double[] outputs = feedForward(inputs);

        double[] outputErrors = new double[outputLayer.length];
        for (int i = 0; i < outputErrors.length; i++) {
            outputErrors[i] = expectedOutputs[i] - outputs[i];
        }

        double[] hiddenErrors = new double[hiddenLayer.length];
        for (int i = 0; i < hiddenErrors.length; i++) {
            hiddenErrors[i] = 0;
            for (int j = 0; j < outputErrors.length; j++) {
                hiddenErrors[i] += outputErrors[j] * weightsHiddenOutput[i][j];
            }
        }

        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            for (int j = 0; j < weightsHiddenOutput[i].length; j++) {
                weightsHiddenOutput[i][j] += learningRate * outputErrors[j] * sigmoidDerivative(outputLayer[j]) * hiddenLayer[i];
            }
        }

        for (int i = 0; i < weightsInputHidden.length; i++) {
            for (int j = 0; j < weightsInputHidden[i].length; j++) {
                weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * sigmoidDerivative(hiddenLayer[j]) * inputs[i];
            }
        }
    }
}