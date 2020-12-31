# MNIST Three/Seven Predictor

This project attempts to classify if a given number from the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is a "3" or a "7".

There are two strategies implemented:
 * The pixel average strategy will average out the pixel intensities of all the 3s and 7s in the training dataset. Numbers in the testing dataset will be compared to see how close they approach these averaged images.
 * The Stochastic Gradient Descent strategy will train a series of weights to value if a certain pixel more often represents a 3 or a 7. The weight adjustment occurs automatically over a series of epochs.

Currently, the SGD strategy doesn't implement back propagation correctly. It will train to identify every number as a 3. The pixel average strategy has an impressive identification rate of 96%.

## Installation and Usage

The project can be built and tested with a simple Maven command.

`mvn clean compile exec:java`

## Contributing

1. Fork it!
2. Create your feature branch: `git checkout -b my-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin my-new-feature`
5. Submit a pull request :D

## Credits

This project idea and algorithm guidelines are from "Deep Learning for Coders with Fastai and PyTorch: AI Applications Without a PhD" by Jeremy Howard and Sylvain Gugger. I made my own implementation in Java to help me solidify the basic training algorithms in my head.

## License

Copyright (c) 2020 Christian Lowe

Released under GNU General Public License v2.0
