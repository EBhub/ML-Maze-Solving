package org;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedList;
import java.util.ListIterator;
import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class Network {

    private INDArray inputLayer;

    /**
     * Does not contain the input layer
     */
    private LinkedList<Layer> layers = new LinkedList<>();
    private LinkedList<Layer> layersReverse = new LinkedList<>();
    private int size;

    public Network(int inputSize) {
        inputLayer = Nd4j.create(new double[inputSize], new int[] { inputSize, 1 });
        size = 1;
    }

    /**
     * Adds a new layer to the ANN
     * 
     * @param size of the new layer
     */
    public void addLayer(int size) {
        int previous;
        if (layers.size() == 0) {
            previous = inputLayer.rows();
        } else {
            previous = layers.getLast().getSize();
        }
        Layer layer = new Layer(size, previous);
        layers.add(layer);
        layersReverse.addFirst(layer);
        size++;
    }

    public static INDArray calculateZ(INDArray weights, INDArray bias, INDArray input) {
        // z'=w*a+b
        return weights.mmul(input).add(bias);
    }

    public INDArray propagate(INDArray input) {
        inputLayer = input;
        ListIterator<Layer> iterator = layers.listIterator();
        Layer l = iterator.next();

        INDArray w1 = l.getIncomingWeights();
        INDArray b1 = l.getBiases();
        l.setZvalues(w1.mmul(inputLayer).add(b1.reshape(b1.columns(), 1)));
        l.setOutput(sigmoid(l.getZvalues()));
        INDArray output = l.getOutput();

        while (iterator.hasNext()) {
            l = iterator.next();
            w1 = l.getIncomingWeights();
            b1 = l.getBiases().transpose();
            l.setZvalues(w1.mmul(output).add(b1.reshape(b1.columns(), 1)));
            l.setOutput(sigmoid(l.getZvalues()));
            output = l.getOutput();
        }
        return layers.getLast().getOutput();
    }

    public void backpropagate(INDArray actual) {
        // deltaOutput=deltaC/deltaA*sigmoid'(z)
        // deltaC/deltaA=(y-a)
        ListIterator<Layer> iterator = layersReverse.listIterator();
        Layer previous = iterator.next();

        previous.setError(previous.getOutput().sub(actual).mul(sigmoidDerivative(previous.getZvalues(), true)));

        // deltaLayer(L)=(transpose(weights(L+1))*delta(L+1))*sigmoid'(z(L))
        while (iterator.hasNext()) {
            Layer l = iterator.next();
            l.setError(previous.getIncomingWeights().transpose().mmul(previous.getError())
                    .mul(sigmoidDerivative(l.getZvalues(), true)));
            previous = l;
        }
    }

    public void update(double learningRate) {
        // w'(L)=w(L)-learningrate*error(L)*transpose(output(L-1))
        // b'(L)=b(L)-learningrate*error(L)
        ListIterator<Layer> iterator = layers.listIterator();
        INDArray input = inputLayer;
        Layer l = iterator.next();

        // Update weights
        INDArray transError = Nd4j.create(l.getError().toDoubleVector(), new int[] { l.getError().size(0), 1 });
        INDArray outputMatrix = Nd4j.create(input.toDoubleVector(), new int[] { 1, input.size(0) });
        INDArray deltaW = transError.mmul(outputMatrix).mul(learningRate);
        l.setIncomingWeights(l.getIncomingWeights().sub(deltaW));

        // Update biases
        l.setBiases(l.getBiases().sub(l.getBiases().mul(learningRate)));

        while (iterator.hasNext()) {
            input = l.getOutput();
            l = iterator.next();
            transError = Nd4j.create(l.getError().toDoubleVector(), new int[] { l.getError().size(0), 1 });
            outputMatrix = Nd4j.create(input.toDoubleVector(), new int[] { 1, input.size(0) });
            deltaW = transError.mmul(outputMatrix).mul(learningRate);
            l.setIncomingWeights(l.getIncomingWeights().sub(deltaW));
        }
    }

    public static INDArray ReLu(INDArray z) {
        double[][] a = new double[z.rows()][1];
        for (int i = 0; i < z.rows(); i++) {
            if (z.getDouble(i) > 0) {
                a[i][0] = z.getDouble(i);
            }
        }
        return Nd4j.create(a);
    }
}
