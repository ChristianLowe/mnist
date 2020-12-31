package io.chrislowe.mnist;

import io.chrislowe.mnist.inference.*;

public class Main {
    public static void main(String[] args) throws Exception {
        NistDataset trainingDataset = getDataset("/train-labels.idx1-ubyte", "/train-images.idx3-ubyte");
        NistDataset testingDataset = getDataset("/t10k-labels.idx1-ubyte", "/t10k-images.idx3-ubyte");

        // Display example output of the training set numbers
        printNumber(trainingDataset.getImagesForLabel(3).get(0));
        printNumber(trainingDataset.getImagesForLabel(7).get(0));

        // SGD Strategy will print it's accuracy/loss while training
        System.out.println("\nTraining via StochasticGradientDescentStrategy:");
        Strategy sgdStrategy = new StochasticGradientDescentStrategy(10);
        sgdStrategy.train(trainingDataset);

        // Pixel averaging strategy doesn't split the training set into validation data
        System.out.println("\nTraining via PixelAverageStrategy");
        Strategy pixelAverageStrategy = new PixelAverageStrategy();
        pixelAverageStrategy.train(trainingDataset);

        // Run pixel averaging strategy against testing data (has around 96% accuracy)
        int trialCount = Math.min(testingDataset.getImagesForLabel(3).size(), testingDataset.getImagesForLabel(7).size());
        System.out.printf("Simulating %d trials...\n", trialCount);
        int threesCorrect = 0, sevensCorrect = 0;
        for (int index = 0; index < trialCount; index++) {
            double[] image;
            boolean isThree;

            image = testingDataset.getImagesForLabel(3).get(index);
            isThree = pixelAverageStrategy.isThree(image);
            if (isThree) threesCorrect++;

            image = testingDataset.getImagesForLabel(7).get(index);
            isThree = pixelAverageStrategy.isThree(image);
            if (!isThree) sevensCorrect++;
        }

        System.out.printf("Threes correctly identified: %d%%; Sevens correctly identified: %d%%\n",
                toPercent(threesCorrect, trialCount), toPercent(sevensCorrect, trialCount));
        System.out.printf("Average across both: %d%%\n",
                toPercent(threesCorrect + sevensCorrect, trialCount * 2));
    }

    private static NistDataset getDataset(String labelsResource, String imagesResource) throws Exception {
        return new NistDataset(labelsResource, imagesResource, Integer.MAX_VALUE);
    }

    private static void printNumber(double[] sample) {
        System.out.println();
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int num = (int) Math.min(sample[i*28 + j] * 10, 9);
                System.out.printf("%01d", num);
            }
            System.out.println();
        }
        System.out.println();
    }

    private static int toPercent(int numerator, int denominator) {
        return (int) ((((double)numerator)/denominator)*100);
    }
}
