# ITU-Challenge-ML5G-PHY
Baseline implementations for the  “Machine Learning Applied to the Physical Layer of Millimeter-Wave MIMO Systems” (ML5G-PHY), which is part of the “ITU AI/ML 5G Challenge"

Details about the challenge and the datasets are avaiable at http://ai5gchallenge.ufpa.br/ and https://www.lasse.ufpa.br/raymobtime/, respectively.

This example use the [S008 dataset](https://nextcloud.lasseufpa.org/s/FQgjXx7r52c7Ww9) from [Raymobtime](https://www.lasse.ufpa.br/raymobtime/)

To process the beams files, download the following files:

Download [Beams_output](https://nextcloud.lasseufpa.org/s/tPb4WmmJgS6gJaR).

Download [csv_info_file](https://nextcloud.lasseufpa.org/s/afpG6qgmRPaJBfw).

To run the processing after already had the files, run the following command:

```bash
# Install all used packages
pip install numpy
``` 

```bash
python filterBeamsOutput.py csvInfoFile.csv indexOfTheAnalyzedUser
```

Example:
```bash
python filterBeamsOutput.py CoordVehiclesRxPerScene_s008.csv 3
```

It is assumed that **Python 3** is used.
