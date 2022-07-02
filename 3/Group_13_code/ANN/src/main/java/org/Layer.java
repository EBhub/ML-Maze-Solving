package org;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Random;

public class Layer {
    private INDArray incomingWeights;
    private INDArray biases;
    private INDArray zvalues;
    private INDArray output;
    private INDArray error;
    private int size;

    private static double maxStartWeight = 2.4;
    private static double minStartWeight = -2.4;

    /**
     * Constructor for layers which depending on the size of this and the previous
     * layer randomly initializes the weights and the biases.
     * 
     * @param size
     * @param previous
     */
    public Layer(int size, int previous) {
        incomingWeights = Nd4j.rand(new int[] { size, previous }).mul(maxStartWeight - minStartWeight)
                .add(minStartWeight);
        biases = Nd4j.rand(new int[] { size });
        this.size = size;
    }

    public void setIncomingWeights(INDArray weights) {
        this.incomingWeights = weights;
    }

    public INDArray getIncomingWeights() {
        return incomingWeights;
    }

    public void setBiases(INDArray biases) {
        this.biases = biases;
    }

    public INDArray getBiases() {
        return biases;
    }

    public int getSize() {
        return size;
    }

    public void setZvalues(INDArray zvalues) {
        this.zvalues = zvalues;
    }

    public INDArray getZvalues() {
        return zvalues;
    }

    public void setOutput(INDArray output) {
        this.output = output;
    }

    public INDArray getOutput() {
        return output;
    }

    public void setError(INDArray error) {
        this.error = error;
    }

    public INDArray getError() {
        return error;
    }
}
