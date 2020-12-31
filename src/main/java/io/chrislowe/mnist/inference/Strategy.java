package io.chrislowe.mnist.inference;

import io.chrislowe.mnist.NistDataset;

public interface Strategy {
    int IMAGE_SIZE = 28 * 28;
    void train(NistDataset dataset);
    boolean isThree(double[] image);
}
