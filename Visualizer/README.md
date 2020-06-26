# Running Raymobtime_visualizer on your machine

It is assumed that Linux is used

1. First of all you'll need to have Blender installed on your machine. Our files are made in Blender 2.79 that can be downloaded in [this link](https://download.blender.org/release/Blender2.79/).
2. After having Blender installed in your machine, you must have a file with the blender modeling of your scenario. We already provide both used in our simulations: Rosslyn and Beijing. If you're using another one, just make sure you have its full path.
3. We provide three models of vehicles for Car, Bus, and Truck in a file named vehicles.blend. Make sure it's in the current folder.
4. At last, you'll need Wireless InSite Simulation Runs. The vehicles positions and rays info are taking from it. We have some available in our [RayMobTime site](https://www.lasse.ufpa.br/raymobtime/)
5. Now, you can choose some options for your simulation: start/ending run, if you want to visualize the rays, what vehicle you want to track and how many rays you wanna display for it. They are made direct in the file `raymobtime_animation.py`, you just need to change start_run (line 24) and end_run (line 25) to your desired interval of runs to simulate, useRays (line 27) to False or True if you wanna see or not the rays animation, user (line 29) to select the desired vehicle to track (0 will track all vehicles with antena) and rays_quantity (line 31) to the number of rays you wanna display per vehicle.
6. With everything set, you'll have to open your CLI and run the following command:

```bash
blender your_scenario.blend -P Raymobtime_visualizer.py your_insite_folder
```

example:
```bash
blender rosslyn.blend -P Raymobtime_visualizer.py ./s008_simulation
```

`your_scenario` is the .blend file with the model of your city and `your_runs_folder` is the location of your Wireless InSite simulation folder data.

You can donwload Wireless InSite simulation folder data [Here](https://nextcloud.lasseufpa.org/s/QKPC23THnn6pez6)

After running the command, Blender should open with your simulation done, where each run is represented in a specific frame of the animation.
