# Running Raymobtime_visualizer on your machine

It is assumed that Linux is used

1. First of all you'll need to have Blender installed on your machine. Our files are made in Blender 2.79 that can be downloaded in [this link](https://download.blender.org/release/Blender2.79/).
2. After having Blender installed in your machine, you must have a file with the blender modeling of your scenario. We already provide both used in our simulations: Rosslyn and Beijing. If you're using another one, just make sure you have its full path.
3. We provide three models of vehicles for Car, Bus, and Truck in a file named vehicles.blend. Make sure it's in the current folder.
4. At last, you'll need Wireless InSite Simulation Runs. The vehicles positions and rays info are taking from it. You can donwload Wireless InSite simulation folder data [Here](https://nextcloud.lasseufpa.org/s/QKPC23THnn6pez6)
5. With everything set, you'll have to open your CLI and run the following command, supposing you're using the provided data:

    ```bash
    blender rosslyn.blend -P Raymobtime_visualizer.py -- ./s008_simulation
    ```

    A generalization of the command would be:

    ```bash
    blender your_scenario.blend -P Raymobtime_visualizer.py -- your_runs_folder
    ```

    - `your_scenario` is the .blend file with the model of your city
    - `your_runs_folder` is the location of your Wireless InSite simulation folder data.

    After running the command, Blender should open with your simulation done, where each run is represented in a specific frame of the animation.

6. Some options are provided to personalize some parameters of your simulation. They are made direct in the file `raymobtime_animation.py`. Are the following:
    - start_run (line 24) and end_run (line 25): change to fit the interval of runs that you want to simulate.
    - useRays (line 27): True will make them visible and False will omit them.
    - user (line 29): to select the vehicle you wanna track the channels. For example, in a simulation with 10 equiped vehicles, you could choose from 1 to 10, picking 0 will track all of them. If useRays is set to False, there'll be no difference in the simulation.
    - rays_quantity (line 31): change to the number of channels you wanna track per pair of antennas, it'll display the best ones. It'll not do anything if useRays is set to False.

[![Simulation Example](http://img.youtube.com/vi/WkIo4cGDYU4/0.jpg)](https://www.youtube.com/watch?v=WkIo4cGDYU4&feature=youtu.be)