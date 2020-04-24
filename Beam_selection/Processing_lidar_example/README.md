# ITU-Challenge-ML5G-PHY
Baseline implementations for the  “Machine Learning Applied to the Physical Layer of Millimeter-Wave MIMO Systems” (ML5G-PHY), which is part of the “ITU AI/ML 5G Challenge"

Details about the challenge and the datasets are avaiable at http://ai5gchallenge.ufpa.br/ and https://www.lasse.ufpa.br/raymobtime/, respectively.

To process the lidar and beams files, download the following files:

Download [PCD](https://nextcloud.lasseufpa.org/s/pwSk9CJnsZoK2ts).

Download [Beams_output](https://nextcloud.lasseufpa.org/s/tPb4WmmJgS6gJaR).

Download [csv_info_file](https://nextcloud.lasseufpa.org/s/afpG6qgmRPaJBfw).

To run the processing after already had the files, run the following command:

```bash
python createInputFromLIDAR.py obstacles_new_3D/ beams_input_user_3.npz
```

```bash
python filterBeamsOutput.py
```

It is assumed that **Python 3** is used.
