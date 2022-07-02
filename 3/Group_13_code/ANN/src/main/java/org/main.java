package org;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.ops.transforms.*;

import java.io.File;
import java.io.FileNotFoundException;

import static org.nd4j.linalg.ops.transforms.Transforms.*;

public class main {
    public static void main(String[] args) throws FileNotFoundException {
        // Input parameters
        double maxStartWeight = 4096;
        double minStartWeight = -4096;
        int[] neuronsPerLayer = new int[] { 10, 10, 7 }; // This array defines the amount of neurons for every layer
        int nLayers = neuronsPerLayer.length; // Amount of layers, including input layer
        double learningrate = 1.0;
        int nEpochs = 2;

        Network n = new Network(neuronsPerLayer[0]);
        for (int i = 1; i < nLayers; i++) {
            n.addLayer(neuronsPerLayer[i]);
        }

        int batchSize = 10;
        DataInputSuperSystem dataInputSuperSystem = new DataInputSuperSystem(new File("./data/features.txt"),
                new File("./data/targets.txt"), 7854);
        INDArray[] dataBatches = dataInputSuperSystem.batchify(dataInputSuperSystem.getDataSet(), batchSize);
        INDArray[] targetBatches = dataInputSuperSystem.batchify(dataInputSuperSystem.getTargetDataSet(), batchSize);
        for (int j = 0; j < 10; j++) {
            for (int i = 0; i < dataInputSuperSystem.getDataSet().rows(); i++) {
                // Set input and actual output
                INDArray input = dataInputSuperSystem.getDataSet().getRow(i).reshape(neuronsPerLayer[0], 1);
                INDArray actual = dataInputSuperSystem.getTargetDataSet().getRow(i)
                        .reshape(neuronsPerLayer[neuronsPerLayer.length - 1], 1);
                INDArray output = Nd4j.create(n.propagate(input).toDoubleVector(), new int[] { 7 });

                // Compute performance
                INDArray loss = loss(output, actual);

                // Compute errors for learning
                n.backpropagate(actual);

                n.update(learningrate);
            }
        }
        int right = 0;
        for (int i = 0; i < dataInputSuperSystem.getTestSet().rows(); i++) {
            INDArray input = dataInputSuperSystem.getTestSet().getRow(i).reshape(neuronsPerLayer[0], 1);
            INDArray actual = dataInputSuperSystem.getTargetTestSet().getRow(i)
                    .reshape(neuronsPerLayer[neuronsPerLayer.length - 1], 1);
            INDArray output = Nd4j.create(n.propagate(input).toDoubleVector(), new int[] { 7 });
            int index1 = 0;
            int index2 = 0;
            for (int j = 1; j < 7; j++) {
                if (output.getDouble(j) > output.getDouble(index1)) {
                    index1 = j;
                }
                if (actual.getDouble(j) > actual.getDouble(index2)) {
                    index2 = j;
                }
            }
            if (index1 == index2) {
                right++;
            }
        }
        System.out.println((double) right / dataInputSuperSystem.getTargetTestSet().rows());

        DataInputSuperSystem unknownData = new DataInputSuperSystem(new File("./data/unknown.txt"), 784);
        String unknownClassified = ""
                + transform(n.propagate(unknownData.getDataSet().getRow(0).reshape(neuronsPerLayer[0], 1)));
        for (int i = 1; i < unknownData.getDataSet().rows(); i++) {
            INDArray input = unknownData.getDataSet().getRow(i).reshape(neuronsPerLayer[0], 1);
            unknownClassified = unknownClassified + ", " + transform(n.propagate(input));
        }
        System.out.println(unknownClassified);
    }

    public static int transform(INDArray arr) {
        int index = 0;
        for (int i = index; i < arr.rows(); i++) {
            if (arr.getDouble(index) < arr.getDouble(i)) {
                index = i;
            }
        }
        return index + 1;
    }

    public static INDArray loss(INDArray output, INDArray actual) {
        // C(w,b)=1/2*(y-a)^2
        return pow(output.sub(actual), 2).muli(0.5);
    }

    public static INDArray computeOutputError(INDArray output, INDArray actual, INDArray z) {
        // deltaOutput=deltaC/deltaA*sigmoid'(z)
        // deltaC/deltaA=(y-a)
        INDArray test = output.sub(actual).mul(sigmoidDerivative(z, true));
        return test;
    }

    public static INDArray backpropagate(INDArray weights, INDArray delta, INDArray z) {
        // deltaLayer(L)=(transpose(weights(L+1))*delta(L+1))*sigmoid'(z(L))
        return weights.transpose().mmul(delta).mul(sigmoidDerivative(z, true));
    }

    public static INDArray updateWeights(double learningRate, INDArray weights, INDArray error, INDArray output) {
        // w'(L)=w(L)-learningrate*error(L)*transpose(output(L-1))
        INDArray transError = Nd4j.create(error.toDoubleVector(), new int[] { error.size(0), 1 });
        INDArray outputMatrix = Nd4j.create(output.toDoubleVector(), new int[] { 1, output.size(0) });
        INDArray deltaW = transError.mmul(outputMatrix).mul(learningRate);
        return weights.sub(deltaW);
    }

    public static INDArray updateBiases(double learningRate, INDArray bias, INDArray error) {
        // b'(L)=b(L)-learningrate*error(L)
        return bias.sub(error.mul(learningRate));
    }
}
