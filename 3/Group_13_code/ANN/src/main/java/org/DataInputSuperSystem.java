package org;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class DataInputSuperSystem {
    private INDArray dataSet;
    private INDArray validationSet;
    private INDArray testSet;

    private INDArray targetDataSet;
    private INDArray targetValidationSet;
    private INDArray targetTestSet;

    private Scanner scData;
    private Scanner scTarget;

    private int size;

    private static float trainingSize = (float) 0.7;
    private static float validationSize = (float) 0.1;
    private static float testSize = (float) 0.2;

    /**
     * Basic constructor to use when training including both data and targets.
     * 
     * @param dataFile   file containing features of the training data.
     * @param targetFile file containing the true labels of the training data.
     * @param size       the size of the files.
     * @throws FileNotFoundException in case file can not be found in the system.
     */
    public DataInputSuperSystem(File dataFile, File targetFile, int size) throws FileNotFoundException {
        scData = new Scanner(dataFile);
        scTarget = new Scanner(targetFile);
        double[][] dataSetTemp = new double[size][10];
        double[][] targetSet = new double[size][7];
        int index = 0;
        while (scData.hasNextLine()) {
            String line = scData.nextLine();
            int target = scTarget.nextInt();
            String[] parts = line.split(",");
            double[] inputVector = new double[10];
            for (int i = 0; i < parts.length; i++) {
                inputVector[i] = Double.parseDouble(parts[i]);
            }
            dataSetTemp[index] = inputVector;
            targetSet[index] = targetMap(target);
            index++;
        }
        dataSet = Nd4j.create(dataSetTemp).get(NDArrayIndex.interval(0, Math.round(trainingSize * size)));
        validationSet = Nd4j.create(dataSetTemp).get(NDArrayIndex.interval(Math.round(trainingSize * size),
                Math.round(trainingSize * size) + Math.round(validationSize * size)));
        testSet = Nd4j.create(dataSetTemp)
                .get(NDArrayIndex.interval(Math.round(trainingSize * size) + Math.round(validationSize * size), size));

        targetDataSet = Nd4j.create(targetSet).get(NDArrayIndex.interval(0, Math.round(trainingSize * size)));
        targetValidationSet = Nd4j.create(targetSet).get(NDArrayIndex.interval(Math.round(trainingSize * size),
                Math.round(trainingSize * size) + Math.round(validationSize * size)));
        targetTestSet = Nd4j.create(targetSet)
                .get(NDArrayIndex.interval(Math.round(trainingSize * size) + Math.round(validationSize * size), size));

        scData.close();
        scTarget.close();
    }

    public static double[] targetMap(int target) {
        switch (target) {
        case 1:
            return new double[] { 1, 0, 0, 0, 0, 0, 0 };
        case 2:
            return new double[] { 0, 1, 0, 0, 0, 0, 0 };
        case 3:
            return new double[] { 0, 0, 1, 0, 0, 0, 0 };
        case 4:
            return new double[] { 0, 0, 0, 1, 0, 0, 0 };
        case 5:
            return new double[] { 0, 0, 0, 0, 1, 0, 0 };
        case 6:
            return new double[] { 0, 0, 0, 0, 0, 1, 0 };
        case 7:
            return new double[] { 0, 0, 0, 0, 0, 0, 1 };
        default:
            return new double[] { 0, 0, 0, 0, 0, 0, 0 };
        }
    }

    /**
     * Overloaded constructor in case of unknown targets to be used when using the
     * network and not training it.
     * 
     * @param dataFile the file containing the unclassified data.
     * @param size     the size of the file.
     * @throws FileNotFoundException in case file can not be found in the system.
     */
    public DataInputSuperSystem(File dataFile, int size) throws FileNotFoundException {
        scData = new Scanner(dataFile);
        double[][] dataSetTemp = new double[size][10];
        double[] targetSet = new double[size];
        int index = 0;
        while (scData.hasNextLine()) {
            String line = scData.nextLine();
            String[] parts = line.split(",");
            double[] inputVector = new double[10];
            for (int i = 0; i < parts.length; i++) {
                inputVector[i] = Double.parseDouble(parts[i]);
            }
            dataSetTemp[index] = inputVector;
            index++;
        }
        dataSet = Nd4j.create(dataSetTemp);
        scData.close();
    }

    public INDArray[] batchify(INDArray dataSet, int batchSize) {
        INDArray[] batches = new INDArray[Math.round(dataSet.rows() / batchSize) - 1];
        for (int i = 0; i < batches.length; i++) {
            batches[i] = dataSet.get(NDArrayIndex.interval(i * batchSize, i * batchSize + batchSize),
                    NDArrayIndex.all());
        }
        return batches;
    }

    public INDArray getDataSet() {
        return dataSet;
    }

    public INDArray getValidationSet() {
        return validationSet;
    }

    public INDArray getTestSet() {
        return testSet;
    }

    public INDArray getTargetDataSet() {
        return targetDataSet;
    }

    public INDArray getTargetValidationSet() {
        return targetValidationSet;
    }

    public INDArray getTargetTestSet() {
        return targetTestSet;
    }
}
