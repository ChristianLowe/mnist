package io.chrislowe.mnist;

import java.io.File;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NistDataset {
    private final Map<Integer, List<double[]>> labelToImagesMap;

    public NistDataset(Map<Integer, List<double[]>> labelToImagesMap) {
        this.labelToImagesMap = labelToImagesMap;
    }

    public NistDataset(String labelsResource, String imagesResource, int limit) throws Exception {
        String labelsFilename = Main.class.getResource(labelsResource).getFile();
        String imagesFilename = Main.class.getResource(imagesResource).getFile();

        List<Integer> labels = extractLabels(new File(labelsFilename), limit);
        List<double[]> images = extractImages(new File(imagesFilename), limit);

        labelToImagesMap = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            int label = labels.get(i);
            double[] image = images.get(i);
            labelToImagesMap.computeIfAbsent(label, k -> new ArrayList<>()).add(image);
        }
    }

    private List<Integer> extractLabels(File labelFile, int limit) throws Exception {
        RandomAccessFile data = getData(labelFile, 0x00000801);
        int count = Math.min(limit, data.readInt());

        List<Integer> labels = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            labels.add(data.readUnsignedByte());
        }

        return labels;
    }

    private List<double[]> extractImages(File imageFile, int limit) throws Exception {
        RandomAccessFile data = getData(imageFile, 0x00000803);
        int count = Math.min(limit, data.readInt());
        int rows = data.readInt();
        int cols = data.readInt();
        int len = rows * cols;

        long lastReportedTime = System.currentTimeMillis();
        System.out.print("Processing images...");

        List<double[]> images = new ArrayList<>(count);
        for (int imageNum = 1; imageNum <= count; imageNum++) {
            long currentTime = System.currentTimeMillis();
            if (imageNum == count || currentTime - lastReportedTime > 250) {
                int percent = (int) ((((float)imageNum)/count) * 100);
                System.out.printf("\rProcessing image %d/%d (%d%%)...", imageNum, count, percent);
                lastReportedTime = currentTime;
            }

            var image = new double[len];
            for (int i = 0; i < len; i++) {
                image[i] = data.readUnsignedByte() / 255.;
            }
            images.add(image);
        }
        System.out.println(" OK.");

        return images;
    }

    private RandomAccessFile getData(File file, int expectedMagicNumber) throws Exception {
        var data = new RandomAccessFile(file, "r");
        int realMagicNumber = data.readInt();
        if (realMagicNumber != expectedMagicNumber) {
            throw new RuntimeException("Input file has invalid magic number.");
        }
        return data;
    }

    public Map<Integer, List<double[]>> getLabelToImagesMap() {
        return labelToImagesMap;
    }

    public List<double[]> getImagesForLabel(int label) {
        return labelToImagesMap.get(label);
    }
}
