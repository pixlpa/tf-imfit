# tf-imfit
Adaptation of the TensorFlow-based rewrite of <https://github.com/mzucker/imfit>
by mzucker.

This version is optimized to work with Google Colab, which was not working with the original repo.
Some effort has been put in to fix deprecated tensorflow functions, but I've been using Tensorflow 1.15 in Colab without issues.

Because this was used to run in a shader in Max, the "makeparams" scripts format the outputs into a json array style that's easy to copy into a dictionary in Max. See the example patcher in the Max folder for details.

I recommend using the Google Colab document for testing this out, and check the original repo for more details.


