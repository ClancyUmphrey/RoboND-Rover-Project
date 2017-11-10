# Do the necessary imports
import argparse
import shutil
import base64
from datetime import datetime
import os
import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO, StringIO
import json
import pickle
import matplotlib.image as mpimg
import time

# Import functions for perception and decision making
from perception import perception_step
from decision import decision_step
from supporting_functions import update_rover, create_output_images
# Initialize socketio server and Flask application 
# (learn more at: https://python-socketio.readthedocs.io/en/latest/)
sio = socketio.Server()
app = Flask(__name__)

# Read in ground truth map and create 3-channel green version for overplotting
# NOTE: images are read in by default with the origin (0, 0) in the upper left
# and y-axis increasing downward.
ground_truth = mpimg.imread('../calibration_images/map_bw.png')
# This next line creates arrays of zeros in the red and blue channels
# and puts the map into the green channel.  This is why the underlying 
# map output looks green in the display image
ground_truth_3d = np.dstack((ground_truth*0, ground_truth*255, ground_truth*0)).astype(np.float)

# Define RoverState() class to retain rover state parameters
class RoverState():
    #===========================================================================
    # Rover properties
    #===========================================================================
    # Throttle      :    -1.0-- 1.0         | <0 reverse, >0 forward
    # Brake         :     0.0-- 1.0         |    both directions
    # Steer angle   :   -15.0--15.0    deg  | <0 right  , >0 left
    # Ground speed  :    -2.0-- 5.0    m/s  | <0 reverse, >0 forward
    # Position      : (0--200, 0--200) m    |         (x, y) 
    # Pitch angle   :
    # Yaw angle     :
    # Roll angle    :
    # Is near object:     Yes--No
    # Is picking up :     Yes--No

    def __init__(self):
        #=======================================================================
        # Incoming From Roversim
        #=======================================================================
        self.start_time = None # To record the start time of navigation
        self.total_time = None # To record total duration of naviagation
        self.img = None # Current camera image
        self.pos = None # Current position (x, y)
        self.yaw = None # Current yaw angle
        self.pitch = None # Current pitch angle
        self.roll = None # Current roll angle
        self.vel = None # Current velocity
        self.near_sample = 0 # Set to telemetry value data["near_sample"]
        self.picking_up = 0 # Set to telemetry value data["picking_up"]
        #=======================================================================
        # Control Parameters 
        #=======================================================================
        self.throttle_set = 0.2 # Setting when increasing speed.
        self.brake_set = 0.5 # Brake setting when braking
        # The stop_forward and go_forward fields below represent total count
        # of navigable terrain pixels.  This is a very crude form of knowing
        # when you can keep going and when you should stop.  Feel free to
        # get creative in adding new fields or modifying these!
        self.stop_forward = 50 # Threshold to initiate stopping
        self.go_forward = 500 # Threshold to go forward again
        self.max_vel = 2.0 # Maximum velocity (meters/second)
        self.view_limit = 6.75 #(m) Limit mapping data to within this distance.
        self.position_tol = 1.0 #(m) Tolerance for considering rover at target.
        self.target_sight = 4.25 #(m) Approximate distance rover can see rocks
        self.y_search_ang_default = 50.0 # degrees to look left while navigating
        self.y_search_ang = self.y_search_ang_default
        # Length of the moving average target buffer (minimum 2)
        self.len_tgt_buf = 3
        # Angle tolerance for looking back when lost
        self.ang_tol = 15.0 # degrees
        self.unstuck_dist = 0.75 #(m) Distance that rover is considered unstuck.
        self.near_rock_tol   = 3.0 #(m)
        #=======================================================================
        # To Update
        #=======================================================================
        #-----------------------------------------------------------------------
        # In Decision
        #-----------------------------------------------------------------------
        self.steer = 0 # Current steering angle
        self.throttle = 0 # Current throttle value
        self.brake = 0 # Current brake value
        self.mode = [] # Current modes
        self.send_pickup = False # Set to True to trigger rock pickup
        self.spot_rock_pos   = None # location when rock spotted
        self.spot_rock_angle = None # angle facing when rock spotted
        self.spot_rock_look  = None # location ahead of rover when rock spotted
        self.picked_up_rocks = np.float64([[],[]])
        self.approach_rock_time = None # for timer in deciding to look for rock
        self.arrived_at_rock_time = -1e-5 # time how long at target
        # Flag to lock rotation direction while avoiding gimbal lock
        self.lock_turn = False 
        # vel_PID Control
        self.P_Verror_prev  = 0.0
        self.time_prev      = 0.0
        self.I_Verror       = 0.0
        # Moving average target buffer
        self.targets = np.zeros((2,self.len_tgt_buf))
        self.lost_time = None
        self.lost_pos = np.zeros(2)
        self.lost_yaw = None
        self.lost_back_angle = None
        self.escape_tgt = np.zeros(2)
        self.start_pos        = None # starting position
        # Look here after starting (in world coords)
        self.start_look       = None 
        # Look angle after starting (in world coords)
        self.start_look_angle = None
        self.wall_start_pos   = None # position of first found wall
        self.wall_time        = None # timer to determine how long at wall
        self.jammed_time = None
        self.jammed_pos  = None 
        self.jammed_look_angle = None
        #-----------------------------------------------------------------------
        # In Perception 
        #-----------------------------------------------------------------------
        self.nav_angles = None # Angles of navigable terrain pixels
        self.nav_dists = None # Distances of navigable terrain pixels
        # Image output from perception step
        # Update this image to display your intermediate analysis steps
        # on screen in autonomous mode
        self.vision_image = np.zeros((160, 320, 3), dtype=np.float) 
        # Worldmap
        # Update this image with the positions of navigable terrain
        # obstacles and rock samples
        self.worldmap = np.zeros((200, 200, 3), dtype=np.float) 
        # Keep score of most likely pixel type (navigable, obstacle, rock)
        self.worldmap_score = np.zeros_like(self.worldmap)
        # Rock location
        self.dist_rock, self.angle_rock = None, None
        self.x_rock, self.y_rock        = None, None
        # Log world rock coordinates
        self.x_rock_world = None
        self.y_rock_world = None
        self.scale = None # scale from rover-centric pixel distance to world
        self.xpix, self.ypix = {}, {} # rover-centric pixel coords
        #=======================================================================
        # Updated Outside perception & decision 
        #=======================================================================
        self.ground_truth = ground_truth_3d # Ground truth worldmap
        self.samples_pos = None # To store the actual sample positions
        self.samples_to_find = 0 # To store the initial count of samples
        self.samples_located = 0 # To store number of samples located on map
        self.samples_collected = 0 # To count the number of samples collected
        

    def is_level(self, tol=1.5):
        """
        Determine if the rover is level, within angle tol, to control if the
        vision information will be used to update the world map.
        """
        tol_m = 360.0-tol
        pitch_level = self.pitch < tol or self.pitch > tol_m
        roll_level   = self.roll < tol or self.roll  > tol_m
        return  pitch_level and roll_level


    def vel_PID(self, Vtarget, P_gain=0.2, I_gain=0.1, D_gain=0.01):
        """
        PID speed control of the rover
        """
        # Set threshold within which the controller doesn't throttle or break.
        throt_thresh =  0.05
        brake_thresh = -0.05

        # Time since last iteration (set to zero on first call).
        if self.time_prev < 1e-14:
            dt = 1e-10
        else:
            dt = self.total_time - self.time_prev

        # PID error terms
        P_Verror  = Vtarget - self.vel
        I_Verror  = np.clip(self.I_Verror + P_Verror * dt, -3.0, 3.0)
        D_Verror  = (P_Verror - self.P_Verror_prev) / dt

        # Save values for next time around.
        self.P_Verror_prev  = P_Verror
        self.time_prev      = self.total_time
        self.I_Verror       = I_Verror
    
        # Acceleration setting: + throttle, - brake.
        accel = (P_gain * P_Verror) + (I_gain * I_Verror) + (D_gain * D_Verror)
        #print('P_Verror ', P_Verror)
        #print('I_Verror ', I_Verror)
        #print('D_Verror ', D_Verror)
        #print('accel ', accel)

        if   accel > throt_thresh:
            self.throttle = np.clip( accel, 0.0, 1.0)
            self.brake    = 0
        elif accel < brake_thresh:
            self.brake    = np.clip(-accel, 0.0, 1.0)
            self.throttle = 0
        else:
            self.brake    = 0
            self.throttle = 0

        if self.vel <= 1e-5:
            # Release brake for turning
            self.brake = 0


    def distance_from(self, world_pos):
        """
        Distance of rover from world_pos.
        """
        return np.linalg.norm(np.array(self.pos) - world_pos)


    def min_distance(self, position, positions):
        """
        Returns the minimum distance of one position from an array of positions.
        """
        delta = positions - position.reshape(2,1)
        dist  = np.sqrt(delta[0]**2 + delta[1]**2)
        return np.amin(dist)


    def world_angle_to(self, position):
        """
        Return the world angle from the rover to position.
        """
        relative_vector = position - np.array(self.pos)
        relative_angle  = np.arctan2(relative_vector[1], relative_vector[0])
        # Convert to degrees and shift range from [-180, +180] in arctan2 to 
        # [0, 360] of the world angle system
        return (np.rad2deg(relative_angle) + 360) % 360


    def is_looking_at(self, angle):
        """
        Is rover looking at angle in world coordinates.
        """
        # To deal with angle discontinuity.
        if angle > 360.0-self.ang_tol or angle < self.ang_tol:
            return self.yaw > 360.0-self.ang_tol or self.yaw < self.ang_tol
        else:
            return angle-self.ang_tol < self.yaw < angle+self.ang_tol


    def set_start_look_AND_angle(self):
        # Determine region of map that the rover started in.
        # Top Region
        a_t   = np.arctan2(4.0, 6.0)     # angle of line dividing top
        P_t   = np.array([91.0, 81.0])  # point on line dividing top
        a_n_t = a_t+np.pi/2.0           # angle of normal
        n_t   = np.array([np.cos(a_n_t), np.sin(a_n_t)]) # normal
        L_t   = n_t # direction to look if in top region
        # Center Region
        a_c     = np.arctan2(8.0, -4.0)   # angle of line dividing center
        P_c     = np.array([101.0, 76.0]) # point on line dividing center
        a_n_c   = a_c-np.pi/2.0           # angle of normal
        n_c     = np.array([np.cos(a_n_c), np.sin(a_n_c)]) # normal
        # Center right
        a_l_c_r = a_c-np.deg2rad(145)         # angle to look if on right
        L_c_r   = np.array([np.cos(a_l_c_r), np.sin(a_l_c_r)]) # direction 
        # Center left
        a_l_c_l = a_c+np.deg2rad(145)         # angle to look if on left
        L_c_l   = np.array([np.cos(a_l_c_l), np.sin(a_l_c_l)]) # direction

        # If above top
        if   np.dot(self.start_pos - P_t, n_t) > 0.0:
            look_dir = L_t
            look_ang = a_n_t
        # If right of center
        elif np.dot(self.start_pos - P_c, n_c) > 0.0:
            look_dir = L_c_r
            look_ang = a_l_c_r
        # If left of center
        else:
            look_dir = L_c_l
            look_ang = a_l_c_l

        # Set a point in world coords in the look direction (200 m away).
        self.start_look = self.start_pos + 200.0 * look_dir

        # Set the angle in the direction of start_look
        self.start_look_angle = np.rad2deg(look_ang) % 360


    def set_I(self, value=0.0):
        self.I_Verror = value


    def slam_on_brakes(self):
        self.set_I(2.0)
        self.vel_PID(0.0, P_gain=100.0, I_gain=-1.0)


    def world2rover(self, pos_world, scale=1.0):
         """
         Transform world postion to rover coordinates.
         Pass in the desired scale (default is to leave in meters)
         """
         # Un-translate
         x = (pos_world[0] - self.pos[0]) * scale
         y = (pos_world[1] - self.pos[1]) * scale
    
         # Un-rotate
         yaw_rad_m = -np.deg2rad(self.yaw)
    
         x_ = (x * np.cos(yaw_rad_m)) - (y * np.sin(yaw_rad_m))
         y_ = (x * np.sin(yaw_rad_m)) + (y * np.cos(yaw_rad_m))
    
         return np.array([x_, y_])
            

    def rover2world(self, pos_rov, scale=1.0):
        """
        Transform rover postion to world coordinates.
        Pass in the desired scale (default is to leave in meters)
        """
        # Rotate
        yaw_rad = np.deg2rad(self.yaw)

        x = (pos_rov[0] * np.cos(yaw_rad)) - (pos_rov[1] * np.sin(yaw_rad))
        y = (pos_rov[0] * np.sin(yaw_rad)) + (pos_rov[1] * np.cos(yaw_rad))

        # Translate
        x = (x / scale) + self.pos[0]
        y = (y / scale) + self.pos[1]

        return np.array([x, y])


    def navigable_target_follow_wall(self, buf=0.70):
        """
        Shift a navigation target forward and in the direction of y_search_a 
        until it is clear of obstacles by the buffer radius.
        target and buf are expected to be in meters.
        Returned target is in meters.
        """
        x_search   = self.target_sight # m
        buf_fract  = 0.25
        nx         = x_search/(buf_fract*buf) + 1
        x_min      = 0.25 # 0.5
        y_search_a = self.y_search_ang #50.0 # degrees
        lat_buf    = 1.15#1.25 # m
        default_y = x_search * np.tan(np.deg2rad( y_search_a ))

        y_obst, x_obst = self.worldmap[:,:,0].nonzero()

        if not x_obst.shape[0]: return np.float64([x_search, default_y]) 

        obst_rov = self.world2rover(np.float64([x_obst, y_obst]))

        # Get the near obstacles
        near_x = 1.5 * x_search
        near_y = max(1.5 * x_search * np.tan(np.deg2rad( y_search_a )),
                     1.5 * x_search)
        near_obst =   (x_min < obst_rov[0]) & (obst_rov[0] < near_x) \
                    & (-near_y < obst_rov[1]) & (obst_rov[1] < near_y)
        obst_rov = obst_rov[:, near_obst]
    
        if not obst_rov.shape[1]: return np.float64([x_search, default_y]) 

        x_range = np.linspace(0.0, x_search, nx)

        # Brute force search for x limit (bisection would be better)
        for i in range(len(x_range)):
            r = np.sqrt( \
                (obst_rov[0]-x_range[i])**2 + (obst_rov[1]-0.0)**2)
            if np.any(r < buf):
                i = max(i-1, 0)
                break
        x_limit = x_range[i]
        
        if x_limit < 1e-10:
            return np.float64([0.0, 0.0])

        y_search = x_limit * np.tan(np.deg2rad( y_search_a ))
        ny = y_search/(buf_fract*buf) + 1
        y_range_l = np.linspace(0.0, y_search, ny)
        y_range_r = -y_range_l

        # y = 0.0 doesn't need checked
        if len(y_range_l) <= 1:
            y_limit_l = 0.0
        else:
            # Brute force search for y limit (bisection would be better)
            for i in range(1,len(y_range_l)): 
                r = np.sqrt( \
                 (obst_rov[0]-x_limit)**2 + (obst_rov[1]-y_range_l[i])**2)
                if np.any(r < buf):
                    i -= 1
                    break
            y_limit_l = y_range_l[i]

        # y = 0.0 doesn't need checked
        if len(y_range_r) <= 1:
            y_limit_r = 0.0
        else:
            # Brute force search for y limit (bisection would be better)
            for i in range(1,len(y_range_r)): 
                r = np.sqrt( \
                 (obst_rov[0]-x_limit)**2 + (obst_rov[1]-y_range_r[i])**2)
                if np.any(r < buf):
                    i -= 1
                    break
            y_limit_r = y_range_r[i]
        
        y_limit = max(y_limit_l - lat_buf, np.average([y_limit_l, y_limit_r]))

        return np.float64([x_limit, y_limit])


    def navigable_world_target(self, target, buf=0.5):
        """
        Shift a navigation target to the nearest navigable location, plus a 
        buffer.
        target and buf are expected to be in meters and world coordinates.
        Returned target is in meters and world coordinates.
        """
        y_navg, x_navg = self.worldmap[:,:,2].nonzero()

        if x_navg.shape[0]:
          r = np.sqrt((target[0]-x_navg)**2 + (target[1]-y_navg)**2)
          I = np.argsort(r)[0]
          if r[I] > 0.0:
            # Vector from target_init to nearest navg
            v_o = np.array([x_navg[I]-target[0], y_navg[I]-target[1]])
            # Distance to shift target towards navg
            d   =  r[I] + buf
            # vector to shift target by
            v_tshift = d * (v_o/r[I]) # unit vector scaled by d
            # Shift target to navg
            target = target + v_tshift
            # Set target to nearest navg
            return target
        return target


    def nearest_rock(self):
        """
        Return nearest rock in world coordinates
        """
        y_rock, x_rock = self.worldmap[:,:,1].nonzero()
        if x_rock.shape[0]:
            r = np.sqrt((x_rock-self.pos[0])**2 + (y_rock-self.pos[1])**2)
            I = np.argsort(r)[0]
       
            return np.array([x_rock[I], y_rock[I]])
        return np.array([0, 0])


    def nearest_rock_not_picked_up(self):
        """
        Return nearest rock not picked up (in world coordinates).
        """
        y_rock, x_rock = self.worldmap[:,:,1].nonzero()
        if x_rock.shape[0]:
            r = np.sqrt((x_rock-self.pos[0])**2 + (y_rock-self.pos[1])**2)
            I = np.argsort(r)
            # Default is the nearest rock.
            i = I[0]
            # Determine if picked up.
            if self.picked_up_rocks.shape[1]:
                for i in I:
                    r = np.sqrt((self.picked_up_rocks[0] - x_rock[i])**2 +
                                (self.picked_up_rocks[1] - y_rock[i])**2 ) 
                    # If more than near_rock_tol meters from a picked up rock:
                    if np.amin(r) > self.near_rock_tol:
                        break
                else:
                    return np.array(self.pos)
            return np.array([x_rock[i], y_rock[i]])
        return np.array(self.pos)


    def is_near_picked_up_rock(self, pos, tol=0):
        """
        Return True if pos is near a picked up rock.
        """
        if tol: near_rock_tol = tol
        else  : near_rock_tol = self.near_rock_tol

        r = np.sqrt((self.picked_up_rocks[0] - pos[0])**2 +
                    (self.picked_up_rocks[1] - pos[1])**2)
        return True if np.any(r < near_rock_tol) else False


    def add_mode(self, mode):
        """
        Add mode to self.mode list if it is not already in the there.
        """
        if not mode in self.mode: self.mode.append(mode)


    def remove_mode(self, mode):
        """
        Remove all occurrences of mode from the self.mode list.
        """
        # Assure list
        if not isinstance(mode, (list, tuple)):
            mode = [mode]
        # Flatten list
        #mode = [element for row in mode for element in row]
        # Remove all elements in list
        self.mode = [ m for m in self.mode if m not in mode ]

    def turn_towards(self, target):
        """
        Turn rover towards target while avoiding problems in the gimbal lock
        zone at +-180 degrees (directly behind rover).
        """
        gimbal_lock_tol = 45.0 # degrees
        gimbal_buffer   = 10.0 # degrees
        steer      = np.rad2deg( np.arctan2(target[1], target[0]) )

        # If in lock zone:
        if abs(steer) > gimbal_lock_tol:
            self.lock_turn = True
        # Release from lock when target is on the right of the lock zone or
        # gimbal_lock_tol away from the lock zone.
        if steer > (-180.0 + gimbal_lock_tol) \
        or abs(steer) > (gimbal_lock_tol + gimbal_buffer):
            self.lock_turn = False
        # Issue steer command
        if self.lock_turn:
            self.steer = -15
        else:
            self.steer = np.clip(steer, -15, 15)
        
    def seek_target(self, target, look=np.float64([0,0])):
        """
        Seek target and turn towards look if arrived at target.
        target and look are positions in rover coordinates.
        Sets self.mode to 'arrived at target' when the rover has done so.
        """
        # If rover is more than position_tol passed the target in the
        # rover-x-direction:
        if target[0] < -self.position_tol:
            self.add_mode('passed target')
            self.remove_mode(['move to target','arrived at target'])

            self.slam_on_brakes()
            self.turn_towards(target)
 
        # If the target is within the position_tol of the rover:
        elif np.linalg.norm(target) < self.position_tol:
            self.add_mode('arrived at target')
            self.remove_mode(['move to target','passed target'])

            self.slam_on_brakes()
            self.turn_towards(look)

        # Else move to target:
        else:
            self.add_mode('move to target')
            self.remove_mode(['arrived at target','passed target'])
            # Linearly decrease velocity as rover approaches target.
            #vel = self.max_vel * (target[0]/self.target_sight)

            # Linearly decrease velocity as rover approaches target and
            # decrease velocity as the angle to the target becomes large.
            t_ang = abs(np.arctan(target[1]/target[0]))
            vel_fact =   (target[0]/self.target_sight) \
                       - (t_ang/self.y_search_ang_default)
            vel = vel_fact * self.max_vel
            # Maintain a minimum velocity.
            vel_min = 0.5 # m/s
            vel     = np.max((vel, vel_min))
            self.vel_PID(vel)
            # Steer
            self.turn_towards(target)


# END class RoverState #########################################################


# Initialize our rover 
Rover = RoverState()

# Variables to track frames per second (FPS)
# Intitialize frame counter
frame_counter = 0
# Initalize second counter
second_counter = time.time()
fps = None


# Define telemetry function for what to do with incoming data
@sio.on('telemetry')
def telemetry(sid, data):

    global frame_counter, second_counter, fps
    frame_counter+=1
    # Do a rough calculation of frames per second (FPS)
    if (time.time() - second_counter) > 1:
        fps = frame_counter
        frame_counter = 0
        second_counter = time.time()
    print("Current FPS: {}".format(fps))

    if data:
        global Rover
        # Initialize / update Rover with current telemetry
        Rover, image = update_rover(Rover, data)

        if np.isfinite(Rover.vel):

            # Execute the perception and decision steps to update the Rover's state
            Rover = perception_step(Rover)
            Rover = decision_step(Rover)

            # Create output images to send to server
            out_image_string1, out_image_string2 = create_output_images(Rover)

            # The action step!  Send commands to the rover!
 
            # Don't send both of these, they both trigger the simulator
            # to send back new telemetry so we must only send one
            # back in respose to the current telemetry data.

            # If in a state where want to pickup a rock send pickup command
            if Rover.send_pickup and not Rover.picking_up:
                send_pickup()
                # Reset Rover flags
                Rover.send_pickup = False
            else:
                # Send commands to the rover!
                commands = (Rover.throttle, Rover.brake, Rover.steer)
                send_control(commands, out_image_string1, out_image_string2)

        # In case of invalid telemetry, send null commands
        else:

            # Send zeros for throttle, brake and steer and empty images
            send_control((0, 0, 0), '', '')

        # If you want to save camera images from autonomous driving specify a path
        # Example: $ python drive_rover.py image_folder_path
        # Conditional to save image frame if folder was specified
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))

    else:
        sio.emit('manual', data={}, skip_sid=True)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control((0, 0, 0), '', '')
    sample_data = {}
    sio.emit(
        "get_samples",
        sample_data,
        skip_sid=True)

def send_control(commands, image_string1, image_string2):
    # Define commands to be sent to the rover
    data={
        'throttle': commands[0].__str__(),
        'brake': commands[1].__str__(),
        'steering_angle': commands[2].__str__(),
        'inset_image1': image_string1,
        'inset_image2': image_string2,
        }
    # Send commands via socketIO server
    sio.emit(
        "data",
        data,
        skip_sid=True)
    eventlet.sleep(0)
# Define a function to send the "pickup" command 
def send_pickup():
    print("Picking up")
    pickup = {}
    sio.emit(
        "pickup",
        pickup,
        skip_sid=True)
    eventlet.sleep(0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()
    
    #os.system('rm -rf IMG_stream/*')
    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("Recording this run ...")
    else:
        print("NOT recording this run ...")
    
    # wrap Flask application with socketio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
