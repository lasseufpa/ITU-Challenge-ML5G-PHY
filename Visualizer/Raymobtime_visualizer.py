'''
To run this code you'll need:
- Blender
- The scenario of your simulation
- The vehicles models named as vehicles.blend in this exact folder
- The Insite simulation of your interest

run as: blender your_scenario.blend -P Raymobtime_visualizer.py Your_InSite_Simulation_Folder
'''
import sys  
import os
import bpy
import csv
import copy
import numpy as np
from bpy import data as D
from bpy import context as C
from mathutils import *
from math import *
from datetime import datetime

def main():    
    #Change values to choose first and last run
    start_run = 0
    end_run = 50
    useRays = True #If you don't want to see the ray simulation, change to False
    user = 3 #Vehicle you wanna track, 0 to track all
    rays_quantity = 4 #Number of rays to be displayed per receiver

    startTime = datetime.now()
    argv = sys.argv
    if len(argv) < 5:
        print('Please, use the following command: blender your_scenario.blend -P blender_animation.py Your_InSite_Simulation_Folder')
        bpy.ops.wm.quit_blender()
        exit(-1)
    #argv = argv[argv.index("--") + 1:]
    #To indicate the input folder position
    folder_scanned_name = argv[4]
    #for key,scene_path in scenes_path.items():
    frame_num = 0
    frame_step = 1
    run = start_run
    D.scenes['Scene'].frame_end = end_run
    D.scenes['Scene'].frame_start = 0
    
    while run <= end_run:
        print('Processing run' + str(run) + ' ...') 
        time_elapsed = datetime.now() - startTime
        scene_path = os.path.join(folder_scanned_name, base_run_dir_fn(run)) 
        
        if not os.path.exists(scene_path):
            print('\nWarning: could not find file ', scene_path , ' Stopping...')
            break
        sumo_info_file = os.path.join(scene_path,'sumoOutputInfoFileName.txt')
        path_info_file = os.path.join(scene_path,'study/model.paths.t001_01.r002.p2m')
        vPosition = getInfoVehicles(sumo_info_file)
        vectorsPath= getInfoPath(path_info_file, user) 
        nVectorsPath = classifyRays(vectorsPath, rays_quantity) 
        animateVehiclesBlender(vPosition,run,frame_step)
        if useRays:
            rayAnimation(nVectorsPath,frame_num,frame_step)
            endRayAnimation(frame_num,frame_step)
        run += 1
        frame_num += frame_step

    endAnimation(frame_num)
    C.scene.frame_set(D.scenes['Scene'].frame_start)
    time_elapsed = datetime.now() - startTime
    print("Total time elapsed: " + str(time_elapsed))

def base_run_dir_fn(i): #the folders will be run00001, run00002, etc.
    """returns the `run_dir` for run `i`"""
    return "run{:05d}".format(i)

def classifyRays(pathInfoList,  numCl=2):
    raysCl = {}
    cleanPathInfo = {}
    newPathInfoList = copy.deepcopy(pathInfoList)

    #clean all None items
    for dirty_ray in pathInfoList.items():
        if not dirty_ray[1]:
            index_dict = dirty_ray[0]
            del newPathInfoList[index_dict]

    for rays in newPathInfoList.items():
        RxLocation = copy.deepcopy(rays[1][len(rays[1])-1])
        dbRx = RxLocation[3]
        RxLocation.pop()
        for i in range(len(RxLocation)):
            RxLocation[i] = str(RxLocation[i])
        key = ' '.join(RxLocation)
        if key in raysCl:
            count += 1
            raysCl[key].append(dbRx)
        else:
            count = 0
            raysCl[key] = [dbRx]
        if count < numCl:
            cleanPathInfo[rays[0]] = rays[1]
            
    return cleanPathInfo

def getInfoPath(path_info_file, Rx_number = 0):
    with open(path_info_file) as pathfile:
        count = 0
        npoints = False
        pathInfoList = {}
        previousLine = ''
        secondLine = ''
        thirdLine = ''
        raysInfoLine = ''
        RxId = 0
        second_l = False
        third_l = False
        RaysOver = True
        RayInfo = 0
        RxRays = 0
        for line in pathfile:
            if(line.startswith('Tx')):
                if RaysOver:
                    Rxinfo = thirdLine.split(' ')
                    try:
                        RxId = int(Rxinfo[0])
                        RxRays = int(Rxinfo[1])
                    except ValueError:
                        RaysOver = False
                    RaysOver = False
                if Rx_number > 0:
                    if RayInfo == RxRays - 1:
                        RaysOver = True
                tmp = line.split('-')
                npoints = len(tmp)
                #ray_number = '%05d' % count
                pathInfoList[count]  = []
                raysInfoLine = previousLine
                RayInfo = int(raysInfoLine.split(' ')[0])
                count += 1
            else:
                if npoints:
                    tmp = line.split(' ')
                    tmp[0] = float(tmp[0])
                    tmp[1] = float(tmp[1])
                    tmp[2] = float(tmp[2])
                    tmp2 = raysInfoLine.split(' ')
                    tmp.append(float(tmp2[2]))
                    if RxId == Rx_number or Rx_number == 0:
                        pathInfoList[count-1].append(tmp)
                    npoints -=1
            if third_l:
                thirdLine = secondLine
            if second_l:
                secondLine = previousLine
                third_l = True
            previousLine = line
            second_l = True
                    
    return pathInfoList


def getInfoVehicles(sumo_info_file):
    #first rotate and then translate
    with open(sumo_info_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='`')
        #line = 0
        vPosition = {}
        for row in reader:
            row['isRx'] = False
            if(row['receiverIndex'] != '-1'):
                row['isRx'] = True
            thisAngleInRad = np.radians(float(row['angle'])) #*np.pi/180
            deltaX = (float(row['length'])/2.0) * np.sin(thisAngleInRad)
            deltaY = (float(row['length'])/2.0) * np.cos(thisAngleInRad)
            vPosition[row['veh']] = {'xinsite':str(float(row['xinsite']) - deltaX),'yinsite':str(float(row['yinsite']) - deltaY),'height':row[' height'],'angle':row['angle'],'isRx':row['isRx'], 'z3':row['z3']}
        
    return vPosition

def createLineBlender(objname, cList, frame_num, frame_step):
    curvedata = D.curves.new(name='curve', type='CURVE')
    curvedata.dimensions = '3D'

    objectdata = D.objects.new(objname, curvedata)
    objectdata.location = (0,0,0) #object origin
    C.scene.objects.link(objectdata)

    polyline = curvedata.splines.new('POLY')
    polyline.points.add(len(cList)-1)
    w = 100

    # Colors
    mat_red = D.materials.new("PKHG")
    mat_red.diffuse_color = (1,0,0)
    mat_blue = D.materials.new("PKHG")
    mat_blue.diffuse_color = (0,0,1)
    mat_green = D.materials.new("PKHG")
    mat_green.diffuse_color = (0,1,0)
    mat_orange = D.materials.new("PKHG")
    mat_orange.diffuse_color = (0.8,0.2,0)
    mat_yellow = D.materials.new("PKHG")
    mat_yellow.diffuse_color = (0.8,0.65,0)
    objectdata.data.materials.append(mat_red)
    for num in range(len(cList)):
        x, y, z, db = cList[num]
        polyline.points[num].co = (x, y, z, w)

    #classify by colors
    if ( db < -220 and db < -193):
        matchoose = mat_blue
    elif ( db < -118):
        matchoose = mat_green
    elif ( db < -100):
        matchoose = mat_yellow
    elif ( db < -95):
        matchoose = mat_orange
    else:
        matchoose = mat_red

    objectdata.active_material = matchoose
    objectdata.data.extrude = 0.1
    objectdata.data.bevel_depth = 0.1

    for i in range(0,frame_num,frame_step):
        C.scene.frame_set(i)
        objectdata.hide = True
        objectdata.hide_render = True
        objectdata.keyframe_insert(data_path="hide_render", index=-1)
        objectdata.keyframe_insert(data_path="hide", index=-1)

    C.scene.frame_set(frame_num)
    objectdata.hide = False
    objectdata.keyframe_insert(data_path="hide", index=-1)

def rayAnimation(vectorsPath,frame_num, frame_step):
    for rays in vectorsPath.items():
        objname = str(frame_num)+'Ray'+str('%05d' % rays[0])
        createLineBlender(objname,rays[1], frame_num, frame_step)


def endRayAnimation(frame_num, frame_step):
    C.scene.frame_set(frame_num + frame_step)
    for x in range(0, len(C.scene.objects)):
        obj_name = C.scene.objects[x].name
        if obj_name.startswith(str(frame_num)): # Add to list
            D.objects[obj_name].hide_render = True
            D.objects[obj_name].hide = True
            D.objects[obj_name].keyframe_insert(data_path="hide_render", index=-1)
            D.objects[obj_name].keyframe_insert(data_path="hide", index=-1)

#Hide all vehicles after last frame/run
def endAnimation(frame_num):
    C.scene.frame_set(frame_num)
    for x in range(0, len(C.scene.objects)):
        obj_name = C.scene.objects[x].name
        if obj_name.startswith('flow'): # Add to list
            D.objects[obj_name].hide_render = True
            D.objects[obj_name].hide = True
            D.objects[obj_name].keyframe_insert(data_path="hide_render", index=-1)
            D.objects[obj_name].keyframe_insert(data_path="hide", index=-1)

#Create and position each vehicle for every frame
def animateVehiclesBlender(vPosition,frame_num,frame_step):
    C.scene.frame_set(frame_num)
    # Pre processamento dos que estao na cena
    objects_in_scene = []
    for x in range(0, len(C.scene.objects)):
        obj_name = C.scene.objects[x].name
        if obj_name.startswith('flow') or obj_name.startswith('dflow') or obj_name.startswith('ped'): # Add to list
            if not obj_name in vPosition:
                D.objects[obj_name].hide_render = True
                D.objects[obj_name].hide = True
                D.objects[obj_name].keyframe_insert(data_path="hide_render", index=-1)
                D.objects[obj_name].keyframe_insert(data_path="hide", index=-1)
                D.objects[obj_name].name = '_'+D.objects[obj_name].name
    for vehicles in vPosition.items():
        if D.objects.get(vehicles[0]) is not None: # Existe, code to move
            veh = D.objects[vehicles[0]]
        else:
            if (float(vehicles[1]['height']) == 1.59): # Car
                bpy.ops.wm.append(directory=os.getcwd().replace('/','//') + "//vehicles.blend/Object/", filepath="vehicles.blend", filename="Car")
                veh = D.objects["Car"]
            elif (float(vehicles[1]['height']) == 3.2): # Bus
                bpy.ops.wm.append(directory=os.getcwd().replace('/','//') + "//vehicles.blend/Object/", filepath="vehicles.blend", filename="Bus")
                veh = D.objects["Bus"]
            elif (float(vehicles[1]['height']) == 4.3): # Truck
                bpy.ops.wm.append(directory= os.getcwd().replace('/','//') + "//vehicles.blend/Object/", filepath="vehicles.blend", filename="Truck")
                veh = D.objects["Truck"]
            elif (float(vehicles[1]['height']) == 0.295): # Drone
                bpy.ops.wm.append(directory= os.getcwd().replace('/','//') + "//vehicles.blend/Object/", filepath="vehicles.blend", filename="Drone")
                veh = D.objects["Drone"]
            else:
                continue
            
            veh.name = vehicles[0]

            # Hide vehicle in frames it's doesn't exist in
            for i in range(0,frame_num,frame_step):
                C.scene.frame_set(i)
                veh.hide_render = True
                veh.hide = True
                veh.keyframe_insert(data_path="hide_render", index=-1)
                veh.keyframe_insert(data_path="hide", index=-1)

        C.scene.frame_set(frame_num)
        veh.hide = False
        veh.hide_render = False
        ax,ay,az = veh.rotation_euler
        angle_to_rotate = 90-float(vehicles[1]['angle'])
        angle_to_rotate = chooseAngleToRotate(degrees(az),angle_to_rotate)
        veh.rotation_euler = (radians(0), radians(0), radians(angle_to_rotate))
        veh.location.xyz = float(vehicles[1]['xinsite']),float(vehicles[1]['yinsite']),float(vehicles[1]['z3'])#float(vehicles[1]['height'])/2 # X,Y,Z
        veh.keyframe_insert(data_path="hide_render", index=-1)
        veh.keyframe_insert(data_path="hide", index=-1)
        veh.keyframe_insert(data_path="location", index=-1)
        veh.keyframe_insert(data_path="rotation_euler", index=-1)


#Choose angle to rotate with minor angle diference with the previous
def chooseAngleToRotate(previousAngle, nextAngle):
    cw = nextAngle - previousAngle 
    ccw = - cw 
    cw360 = convert360(cw)
    ccw360 = convert360(ccw)
    if ( cw360 < ccw360 ) :
        return previousAngle + cw360
    else:
        return previousAngle - ccw360
    
def convert360(x):
    if ( x < 0 ) :
        n = ceil(-x / 360)
        x = x + n*360

    return x % 360

if __name__ == '__main__':
    main()
