package io.chrislowe.mnist;

import java.util.AbstractMap;

// It's 2020 and Java doesn't come with a Pair class -_-
public class Pair<K, V> extends AbstractMap.SimpleEntry<K, V> {
    public Pair(K key, V value) {
        super(key, value);
    }
}
