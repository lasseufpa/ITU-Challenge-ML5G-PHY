# ITU-Challenge-ML5G-PHY
Baseline implementations for the  “Machine Learning Applied to the Physical Layer of Millimeter-Wave MIMO Systems” (ML5G-PHY), which is part of the “ITU AI/ML 5G Challenge"

Details about the challenge and the datasets are avaiable at http://ai5gchallenge.ufpa.br/ and https://www.lasse.ufpa.br/raymobtime/, respectively.

This example use the [S008 dataset](https://nextcloud.lasseufpa.org/s/FQgjXx7r52c7Ww9) from [Raymobtime](https://www.lasse.ufpa.br/raymobtime/)

To process the image files, download the following files:

Download [Camera_images](https://nextcloud.lasseufpa.org/s/Q6tqZt2oAKPToZo).

Download [csv_info_file](https://nextcloud.lasseufpa.org/s/bZi69AXBjJ8ExgA).

```bash
# Install all used packages
pip install numpy opencv-python
``` 

To run the processing after already had the files, run the following command:

```bash
python createInputFromCameras.py CameraFolder csvInfoFile.csv indexOfTheAnalyzedUser
```

Example:
```bash
python createInputFromCameras.py image_data_s008 CoordVehiclesRxPerScene_s008.csv 3
```

It is assumed that **Python 3** is used.

