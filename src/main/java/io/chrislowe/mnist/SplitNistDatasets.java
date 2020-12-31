package io.chrislowe.mnist;

import java.util.*;

public class SplitNistDatasets {
    private static final long RANDOM_SEED = 42;

    private final NistDataset trainingDataset;
    private final NistDataset validationDataset;

    public SplitNistDatasets(NistDataset dataset, int validationFrequency) {
        Random random = new Random(RANDOM_SEED);
        for (List<double[]> images : dataset.getLabelToImagesMap().values()) {
            Collections.shuffle(images, random);
        }

        Map<Integer, List<double[]>> labelToImagesMap = dataset.getLabelToImagesMap();
        Map<Integer, List<double[]>> trainingMap = new HashMap<>();
        Map<Integer, List<double[]>> validationMap = new HashMap<>();
        for (int label : labelToImagesMap.keySet()) {
            trainingMap.put(label, new ArrayList<>());
            validationMap.put(label, new ArrayList<>());

            List<double[]> images = labelToImagesMap.get(label);
            for (int index = 0; index < images.size(); index++) {
                double[] image = images.get(index);
                if (index % validationFrequency != 0) {
                    trainingMap.get(label).add(image);
                } else {
                    validationMap.get(label).add(image);
                }
            }
        }

        trainingDataset = new NistDataset(trainingMap);
        validationDataset = new NistDataset(validationMap);
    }

    public NistDataset getTrainingDataset() {
        return trainingDataset;
    }

    public NistDataset getValidationDataset() {
        return validationDataset;
    }
}
