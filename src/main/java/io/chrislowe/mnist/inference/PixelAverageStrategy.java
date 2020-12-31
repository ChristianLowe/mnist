package io.chrislowe.mnist.inference;

import io.chrislowe.mnist.NistDataset;

import java.util.List;
import java.util.stream.IntStream;

public class PixelAverageStrategy implements Strategy {
    private final double[] averageThrees = new double[IMAGE_SIZE];
    private final double[] averageSevens = new double[IMAGE_SIZE];

    @Override
    public void train(NistDataset dataset) {
        System.out.println("[PixelAverageStrategy] Starting training...");
        long startTime = System.currentTimeMillis();

        calculateAverageImage(averageThrees, dataset.getImagesForLabel(3));
        calculateAverageImage(averageSevens, dataset.getImagesForLabel(7));

        long endTime = System.currentTimeMillis();
        System.out.printf("[PixelAverageStrategy] Training finished in %d ms.\n", endTime - startTime);
    }

    @Override
    public boolean isThree(double[] image) {
        return meanSquaredError(image, averageThrees) < meanSquaredError(image, averageSevens);
    }

    private void calculateAverageImage(double[] averageImage, List<double[]> images) {
        for (double[] image : images) {
            for (int i = 0; i < IMAGE_SIZE; i++) {
                averageImage[i] += image[i] / images.size();
            }
        }
    }

    @SuppressWarnings("OptionalGetWithoutIsPresent")
    private double meanSquaredError(double[] inputImage, double[] averageImage) {
        double squaredError = IntStream.range(0, IMAGE_SIZE)
                .mapToDouble(i -> inputImage[i] - averageImage[i])
                .map(error -> error * error)
                .average()
                .getAsDouble();
        return Math.sqrt(squaredError);
    }
}
