package io.chrislowe.mnist.inference;

import io.chrislowe.mnist.NistDataset;
import io.chrislowe.mnist.Pair;
import io.chrislowe.mnist.SplitNistDatasets;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class StochasticGradientDescentStrategy implements Strategy {
    private static final double LEARN_RATE = 0.3;
    private static final int BATCH_SIZE = 20;
    private static final int EPOCHS = 5;

    private final int validationFrequency;
    private final double[] threeWeights;
    private double bias;

    public StochasticGradientDescentStrategy(int validationFrequency) {
        this.validationFrequency = validationFrequency;
        this.threeWeights = initializeWeights(IMAGE_SIZE);
        this.bias = initializeWeights(1)[0];
    }

    @Override
    public void train(NistDataset dataset) {
        SplitNistDatasets splitNistDatasets = new SplitNistDatasets(dataset, validationFrequency);
        NistDataset trainingDataset = splitNistDatasets.getTrainingDataset();
        NistDataset validationDataset = splitNistDatasets.getValidationDataset();

        List<Pair<Integer, double[]>> trainingPairs = allPairs(trainingDataset.getImagesForLabel(3), trainingDataset.getImagesForLabel(7));
        List<Pair<Integer, double[]>> validationPairs = allPairs(validationDataset.getImagesForLabel(3), validationDataset.getImagesForLabel(7));
        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            long startTime = System.currentTimeMillis();
            System.out.printf("[StochasticGradientDescentStrategy] Epoch %d/%d... ", epoch, EPOCHS);
            Collections.shuffle(trainingPairs);

            // Train the weights in mini-batches
            int batchCount = 0;
            int batchStartIndex = 0;
            double trainingLoss = 0;
            while (batchStartIndex < trainingPairs.size()) {
                int batchEndIndex = Math.min(batchStartIndex + BATCH_SIZE, trainingPairs.size());
                var miniBatch = trainingPairs.subList(batchStartIndex, batchEndIndex);
                double loss = mnistLoss(miniBatch);
                updateParameters(loss);
                batchCount++;
                trainingLoss += loss;
                batchStartIndex = batchEndIndex;
            }

            // Find the loss and accuracy against the validation set
            double correctPercent = 0;
            double validationLoss = mnistLoss(validationPairs);
            for (var validationPair : validationPairs) {
                int label = validationPair.getKey();
                boolean isThree = isThree(validationPair.getValue());
                if (isThree && label == 3 || !isThree && label == 7) {
                    correctPercent += 1 * (1. / validationPairs.size());
                }
            }

            // Report the results of the epoch
            double averageTrainingLoss = trainingLoss / batchCount;
            long timeTaken = System.currentTimeMillis() - startTime;
            System.out.printf("done in %d ms. Batch count: %d, accuracy: %.03f, loss: %.03f, training loss: %.03f\n",
                    timeTaken, batchCount, correctPercent, validationLoss, averageTrainingLoss);
        }
    }

    @Override
    public boolean isThree(double[] image) {
        return sigmoid(score(image)) < 0.5;
    }

    private double[] initializeWeights(int count) {
        return ThreadLocalRandom.current().doubles(count, -1, 1).toArray();
    }

    private double score(double[] image) {
        return IntStream.range(0, IMAGE_SIZE).mapToDouble(i -> image[i] * threeWeights[i]).sum() + bias;
    }

    private double sigmoid(double value) {
        return 1 / (1 + Math.exp(-value));
    }

    private List<Pair<Integer, double[]>> allPairs(List<double[]> threes, List<double[]> sevens) {
        return Stream.concat(
                threes.stream().map(img -> new Pair<>(3, img)),
                sevens.stream().map(img -> new Pair<>(7, img)))
                .collect(Collectors.toList());
    }

    private double mnistLoss(List<Pair<Integer, double[]>> labelToImageList) {
        return labelToImageList.stream()
                .map(p -> new Pair<>(p.getKey(), sigmoid(score(p.getValue()))))
                .mapToDouble(p -> p.getKey().equals(3) ? p.getValue() : 1 - p.getValue())
                .average()
                .orElseThrow();
    }

    private void updateParameters(double loss) {
        double adjustment = loss * LEARN_RATE;
        bias -= adjustment;
        for (int index = 0; index < IMAGE_SIZE; index++) {
            threeWeights[index] -= adjustment;
        }
    }
}
