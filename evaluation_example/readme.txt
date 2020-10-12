The files in this folder show an example of an evaluation of the predictions of
a trained model. It uses the submission example provided in this repository
(i.e., folder named "submission_baseline_example"). So, it is required to run
the submission example before running evaluate_topK.py in this folder. 

To evaluate the predictions of a model, the file equivChannel_beamIndex_gen.py
must be invoked to generate the expected predictions for the training and
testing datasets (s008 and s009, respectively). You only need to do this step
once. You can do with the following command: "python equivChannel_beamIndex_gen.py".
You need to change the dataset path to reflect its location in your system.

Then, you can run the evaluation script "python evaluate_topK.py", which is
configured with default parameters to evaluate the predictions of the submission
example. The script evaluate_topK.py also accepts parameters, which allow
specifying different predictions and true labels. See its help message: 

optional arguments:
  -h, --help            show this help message and exit
  --beam_test_label BEAM_TEST_LABEL
                        Ground truth file
  --beam_test_pred BEAM_TEST_PRED
                        Predictions file
