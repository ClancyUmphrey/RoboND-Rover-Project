import numpy as np


# This is where you can build a decision tree for determining throttle, brake and steer 
# commands based on the output of the perception_step() function
def decision_step(Rover):
    """
    Implement conditionals to decide what to do given perception data.

    (if statements are organized in a hierarchical override system with modes
     farther down possibly having higher rank)
    """
    zero = 1e-10 # zero for floating point comparison

    #-----------------------------------------------------------------------
    # Set default target and look direction.
    #-----------------------------------------------------------------------
    # These defaults may be overridden by the hierarchical modes that follow. 
    look_rov   = np.float64([0.0, -1.0]) # look right when arrived at target
    new_target = Rover.navigable_target_follow_wall()

    #===========================================================================
    # Initialization 
    #===========================================================================
    #---------------------------------------------------------------------------
    # Start Mode
    #---------------------------------------------------------------------------
    # Start by looking at the wall (away from the rocks in the center).

    # Determine if start.
    #---------------------------------------------------------------------------
    # Needs a time delay of 0.01 due to initialization issues.
    if  0.01 < Rover.total_time < 1.0 \
    and not 'start' in Rover.mode:
        Rover.add_mode('start')
        Rover.start_pos  = np.array(Rover.pos)
        Rover.set_start_look_AND_angle()

    # Determine if passed start: if so, find wall
    #---------------------------------------------------------------------------
    if 'start' in Rover.mode \
    and Rover.is_looking_at(Rover.start_look_angle): 
        Rover.remove_mode('start')
        Rover.add_mode('find wall')
        Rover.y_search_ang = 0.0 # drive straight towards wall

    # Action if start.
    #---------------------------------------------------------------------------
    if 'start' in Rover.mode:
        new_target = np.zeros(2)
        look_rov   = Rover.world2rover(Rover.start_look)
        
    #---------------------------------------------------------------------------
    # Find Wall Mode
    #---------------------------------------------------------------------------

    # Determine if found wall
    #---------------------------------------------------------------------------
    if 'find wall' in Rover.mode and 'arrived at target' in Rover.mode:
        # Start timer.
        if Rover.wall_time < 0:
            Rover.wall_time = Rover.total_time
        if Rover.total_time - Rover.wall_time > 2.0: # 2.0 seconds
            Rover.remove_mode('find wall')
            Rover.add_mode('map and pickup')
            Rover.wall_start_pos = np.array(Rover.pos)
            Rover.y_search_ang   = Rover.y_search_ang_default # resume default
    else:
        # Reset timer.
        Rover.wall_time = -1e-5

    #===========================================================================
    # Pickup Rocks
    #===========================================================================
    #---------------------------------------------------------------------------
    # Pickup Rock Mode
    #---------------------------------------------------------------------------

    # Determine if pickup.
    #---------------------------------------------------------------------------
    # If not currently picking up the rock and one is seen, set mode to pickup.
    # (won't be set True until Rover.send_pickup is set to True).
    if  not 'pickup'         in Rover.mode \
    and not 'return to spot' in Rover.mode \
    and     'map and pickup' in Rover.mode \
    and not Rover.picking_up \
    and not Rover.is_near_picked_up_rock(np.array(
                                         [Rover.x_rock, Rover.y_rock])) \
    and Rover.dist_rock > zero:
        Rover.add_mode('pickup') # Flag rock for pickup
        Rover.spot_rock_pos   = np.array(Rover.pos)
        Rover.spot_rock_angle = Rover.yaw
        Rover.spot_rock_look  = np.array(Rover.pos) + \
                                3 * np.array([np.cos(np.deg2rad(Rover.yaw)),
                                              np.sin(np.deg2rad(Rover.yaw))])

    # Determine if done with pickup.
    #---------------------------------------------------------------------------
    # If in a state where we want to pickup a rock, send pickup command.
    if Rover.near_sample and Rover.vel == 0 and not Rover.picking_up:
        Rover.send_pickup = True
        Rover.remove_mode(['pickup', 'approach rock', 'look for rock'])
        Rover.add_mode('look at spot')
        Rover.add_mode('return to spot')
        Rover.picked_up_rocks = np.append(Rover.picked_up_rocks, \
                              Rover.nearest_rock_not_picked_up().reshape(2,1), \
                              axis=1) 

    # Action if pickup.
    #---------------------------------------------------------------------------
    # If a rock has been flagged for pickup, execute steps in order to
    # reach a state for picking it up.
    if 'pickup' in Rover.mode:

        # Determine if approach rock.
        #-----------------------------------------------------------------------
        # If arrived at target and the rover is stopped and sees the rock or
        # is arrived at target for five seconds:
        if  'arrived at target' in Rover.mode \
        and Rover.vel <= zero \
        and (Rover.dist_rock > zero 
             or Rover.total_time - Rover.arrived_at_rock_time > 5.0):
            if not 'approach rock' in Rover.mode:
                Rover.approach_rock_time = Rover.total_time
            Rover.add_mode('approach rock')

        # Time how long at target.
        if 'arrived at target' in Rover.mode:
            # Start timer.
            if Rover.arrived_at_rock_time < 0:
                Rover.arrived_at_rock_time = Rover.total_time
        else:
            # Reset timer.
            Rover.arrived_at_rock_time = -1e-5

        #-----------------------------------------------------------------------
        # Set target (in meters)
        #-----------------------------------------------------------------------

        # Target based on rover vision
        #-----------------------------------------------------------------------
        if 'approach rock' in Rover.mode: 

            # Determine if look for rock.
            #-------------------------------------------------------------------
            if  'arrived at target' in Rover.mode \
            and Rover.vel <= zero \
            and Rover.total_time - Rover.approach_rock_time > 3.0:
                Rover.add_mode('look for rock')

            # Action if look for rock and one is not seen.
            #-------------------------------------------------------------------
            if 'look for rock' in Rover.mode and not Rover.dist_rock > zero:
                look_rov   = np.float64([0.0, 1.0]) # look left
                new_target = np.zeros(2)

            elif Rover.dist_rock > zero:
                # Switch to rover vision coords if rock still in view
                look_rov = np.array([Rover.x_rock, Rover.y_rock]) / Rover.scale
                # Target rock now
                new_target = look_rov
            else:
                # Use nearest rock from world map
                look_rov = Rover.world2rover(Rover.nearest_rock_not_picked_up())
                # Target rock now
                new_target = look_rov

        # Target based on world map
        #-----------------------------------------------------------------------
        else:
            # In world coordinates
            rock   = Rover.nearest_rock_not_picked_up()
            target = Rover.navigable_world_target(rock)
            # In rover coordinates
            new_target = Rover.world2rover(target)
            look_rov   = Rover.world2rover(rock)

        # Target when arrived at sample
        #-----------------------------------------------------------------------
        if Rover.near_sample:
            print('arrived at sample')
            new_target = np.zeros(2)
            look_rov   = np.zeros(2)


    #---------------------------------------------------------------------------
    # Return to Spot Mode
    #---------------------------------------------------------------------------

    # Determine if returned to spot.
    #---------------------------------------------------------------------------
    if  'return to spot' in Rover.mode \
    and Rover.distance_from(Rover.spot_rock_pos) < 2.0 \
    and Rover.is_looking_at(Rover.spot_rock_angle):
        Rover.remove_mode('return to spot')

    # Action if return to spot.
    #---------------------------------------------------------------------------
    if 'return to spot' in Rover.mode:
        # Determine if looking at spot
        #-----------------------------------------------------------------------
        if  'look at spot' in Rover.mode \
        and Rover.is_looking_at(Rover.world_angle_to(Rover.spot_rock_pos)):
            Rover.remove_mode('look at spot')

        # Action if look at spot
        #-----------------------------------------------------------------------
        if 'look at spot' in Rover.mode:
            new_target = np.zeros(2)
            look_rov   = Rover.world2rover(Rover.spot_rock_pos)

        else:
            new_target = Rover.world2rover(Rover.spot_rock_pos)
            look_rov   = Rover.world2rover(Rover.spot_rock_look)

    #===========================================================================
    # Mapping Complete
    #===========================================================================

    #---------------------------------------------------------------------------
    # Back to Start Mode 
    #---------------------------------------------------------------------------

    # Determine if map loop closed. 
    #---------------------------------------------------------------------------
    if  not 'back to start' in Rover.mode \
    and Rover.total_time > 10 * 60 \
    and Rover.distance_from(Rover.wall_start_pos) < 2.5:
        Rover.add_mode('back to start')
        Rover.add_mode('look at start')

    # Determine if back at start.
    #---------------------------------------------------------------------------
    if 'back to start' in Rover.mode \
    and Rover.distance_from(Rover.start_pos) <= Rover.position_tol:
        Rover.remove_mode('back to start')
        Rover.final_time = Rover.total_time
        Rover.add_mode('Finished!')
        Rover.add_mode('Final time: '+str(int(Rover.final_time)))

    # Action if back to start.
    #---------------------------------------------------------------------------
    if 'back to start' in Rover.mode:
        new_target = Rover.world2rover(Rover.start_pos)

        # Determine if looking at start
        #-----------------------------------------------------------------------
        if  'look at start' in Rover.mode \
        and Rover.is_looking_at(Rover.world_angle_to(Rover.start_pos)):
            Rover.remove_mode('look at start')

        # Action if look at start
        #-----------------------------------------------------------------------
        if 'look at start' in Rover.mode:
            new_target = np.zeros(2)
            look_rov   = Rover.world2rover(Rover.start_pos)

        else:
            new_target = Rover.world2rover(Rover.start_pos)

    #===========================================================================
    # Get Unstuck
    #===========================================================================
    #---------------------------------------------------------------------------
    # Lost Mode
    #---------------------------------------------------------------------------

    # Determine if lost.
    #---------------------------------------------------------------------------
    if  Rover.vel <= zero \
    and Rover.throttle <= zero \
    and 'arrived at target' in Rover.mode \
    and not 'lost' in Rover.mode:
        Rover.add_mode('lost')
        Rover.lost_time      = Rover.total_time
        Rover.lost_pos       = Rover.pos
        Rover.lost_yaw       = Rover.yaw
        # angle looking behind
        Rover.lost_back_angle = (Rover.lost_yaw + 180.0) % 360

    # Determine if unlost (unstuck_dist m away from lost_pos).
    #---------------------------------------------------------------------------
    if Rover.distance_from(Rover.lost_pos) > Rover.unstuck_dist:
        Rover.remove_mode('lost')
        Rover.lost_pos = np.zeros(2)

    # Action if lost.
    #---------------------------------------------------------------------------
    if 'lost' in Rover.mode and Rover.total_time - Rover.lost_time > 10.0:
        # Revert to rover vision navigation if turned around after lost.
        if  Rover.is_looking_at(Rover.lost_back_angle) \
        and len(Rover.nav_angles) >= Rover.go_forward:
            print('Un-sticking -----------------------------------------------')
            nav_angle  = np.mean(Rover.nav_angles)
            dist       = Rover.position_tol * 1.5
            new_target = \
                np.array([dist*np.cos(nav_angle), dist*np.sin(nav_angle)])

    #---------------------------------------------------------------------------
    # Jammed Mode
    #---------------------------------------------------------------------------

    # Determine if jammed.
    #---------------------------------------------------------------------------
    if  Rover.vel <= zero \
    and Rover.throttle >= zero \
    and not 'arrived at target' in Rover.mode \
    and not 'jammed' in Rover.mode:
        Rover.add_mode('jammed')
        Rover.jammed_time = Rover.total_time
        Rover.jammed_pos  = Rover.pos
        Rover.jammed_look_ang = (Rover.yaw - 45.0) % 360

    # Determine if un-jammed.
    #---------------------------------------------------------------------------
    if  'jammed' in Rover.mode \
    and (Rover.is_looking_at(Rover.jammed_look_ang)
         or Rover.distance_from(Rover.jammed_pos) > Rover.unstuck_dist):
        Rover.remove_mode('jammed')

    # Action if jammed.
    #---------------------------------------------------------------------------
    if  'jammed' in Rover.mode and Rover.total_time - Rover.jammed_time > 5.0:
        print('Un-jamming ---------------------------------------------------')
        new_target = np.zeros(2)
        look_rov   = np.float64([0.0, -1.0]) # look right 
    
    #===========================================================================
    # Finished Mode
    #===========================================================================
    if 'Finished!' in Rover.mode:
        new_target = np.zeros(2)

    #===========================================================================
    # Seek target
    #===========================================================================
    # Use moving average target to smooth the control
    # Cycle in a new target (also resets targets length if modified)
    Rover.targets = \
        np.append(Rover.targets[:,1-Rover.len_tgt_buf:],
                  new_target.reshape(2,1), axis=1)

    target_rov = np.average(Rover.targets, axis=1)

    Rover.seek_target(target_rov, look_rov)

    print('TARGET ',target_rov)
    print('MODE ', Rover.mode)

    return Rover

# END OF FILE ##################################################################
