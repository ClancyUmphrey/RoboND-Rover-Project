import numpy as np
import cv2
import matplotlib.image as mpimg

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):
           
    M = cv2.getPerspectiveTransform(src, dst)
    # keep same size as input image
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    
    return warped


def within_view(img, src, dst):
    """
    Identify pixels that are within view of the rover in the perspective
    transform, so they can be included for the obstacle threshold.  This
    is needed when the threshold is applied after the perspective transform,
    as is the case in this project.
    
    Args:
        img (np.uint8): Sample RGB image on which perspective transform will
            be applied.
        src (np.float32): Array of four source points for the perspective
            transform.
        dst (np.float32): Array of four destination points for the perspective
            transform.
            
    Returns:
        (np.bool): Boolean array of shape (img.shape[0], img.shape[1]) that is
            True for each pixel within view.
    """
    RGB_max = 255
    white_img = np.full_like(img, RGB_max)
    warped = perspect_transform(white_img, src, dst)
    # All the areas that are RGB_max are within view.
    return warped[:,:,0] == RGB_max


def get_hsv_range(img, scale=1.0):
    """
    Given a calibration image(s) that includes only the object and white space
    everywhere else, return lower and upper HSV thresholds based on the
    average and standard deviation of the objects HSV values.
    
    Args:
        img (np.uint8): RGB image, or a tuple or list of images.
        scale (float) : Scale factor for the HSV standard deviation range.
    
    Returns:
        lower (np.uint8): lower HSV threshold.
        upper (np.uint8): upper HSV threshold.
    """
    def get_masked_hsv(img):
        # Get a mask that keeps all of the image that isn't white
        RGB_max = 255
        mask = np.bool(True)
        for i in range(img.shape[2]):
            mask = np.logical_and(mask, img[:,:,i] != RGB_max)

        # Get HSV representation of the image and mask it (which puts the pixels
        # of the object into a column of [H, S, V] elements)
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[mask]

    # Is img a list (or tuple) of multiple images?
    if isinstance(img, (list, tuple)):
        hsv = get_masked_hsv(img[0])
        for imgI in img[1:-1]:
            hsv = np.concatenate((hsv, get_masked_hsv(imgI)))
    else:
        hsv = get_masked_hsv(img)

    # Get the average H, S, and V
    hsv_avg = np.average(hsv, axis=0)

    # Get the standard deviation of H, S, and V
    hsv_std = np.std(hsv, axis=0)

    # Set the HSV range
    lower = np.clip( hsv_avg - scale * hsv_std, 0, 255 ).astype(np.uint8)
    upper = np.clip( hsv_avg + scale * hsv_std, 0, 255 ).astype(np.uint8)

    return lower, upper


def color_thresh(img, rock_hsv_range, in_view, rgb_thresh=(160, 160, 160)):
    """
    Identify pixels that are navigable terrain, obstacles, and rock samples.

    Args:
        img (np.uint8): RGB image.
        rock_hsv_range (tuple, np.uint8): HSV range for identifying rocks.
            Format -- lower_hsv, upper_hsv.
        in_view (np.bool): Boolean array of shape (img.shape[0], img.shape[1])
            that is True for each pixel within view.
        rgb_thresh (tuple, uint8, optional): RGB values that separate navigable
            terrain and obstacles. Default is RGB > 160 for navigable terrain.

    Returns:
        navg_select (np.uint8): Binary image of the navigable pixels.
        obst_select (np.uint8): Binary image of the obstacle pixels.
        rock_select (np.uint8): Binary image of the rock pixels.
    """
    #--------------------------------------------------------------------------
    # Find Ground and Obstacles
    #--------------------------------------------------------------------------
    # Create arrays of zeros same xy size as img, but single channel
    navg_select = np.zeros_like(img[:,:,0])
    obst_select = np.zeros_like(img[:,:,0])
    # Require that each ground pixel be above all three threshold values in RGB.
    # above_thresh will contain a boolean array with "True" where threshold was
    # met.
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                 & (img[:,:,1] > rgb_thresh[1]) \
                 & (img[:,:,2] > rgb_thresh[2])
    # Index the navg array of zeros with the boolean array and set to 1.
    navg_select[above_thresh] = 1
    # Obstacles are the pixels below the threshold and within view.
    obst_select[np.logical_not(above_thresh) & in_view] = 1

    #--------------------------------------------------------------------------
    # Find Rocks 
    #--------------------------------------------------------------------------
    # Convert to HSV space
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Find pixels in rock_hsv_range.
    rock_select = cv2.inRange(hsv, rock_hsv_range[0], rock_hsv_range[1])
    # inRange returns array of 0->False, and 255->True, so clip 255 to 1.
    rock_select = np.clip(rock_select, 0, 1)

    #--------------------------------------------------------------------------
    # Return the binary images
    #--------------------------------------------------------------------------
    return navg_select, obst_select, rock_select


def color_thresh_hsv(img, in_view, \
                     rock_hsv_range, navg_hsv_range, obst_hsv_range):
    """
    Identify pixels that are navigable terrain, obstacles, and rock samples
    using an HSV range for each.

    Args:
        img (np.uint8): RGB i3mage.
        *_hsv_range (tuple, np.uint8): HSV range for identifying image items.
            (*rock, navg, obst).
            Format -- lower_hsv, upper_hsv.

    Returns:
        navg_select (np.uint8): Binary image of the navigable terrain pixels.
        obst_select (np.uint8): Binary image of the obstacle pixels.
        rock_select (np.uint8): Binary image of the rock pixels.
    """
    # Convert to HSV space
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #--------------------------------------------------------------------------
    # Find Navigable Terrain, Obstacles, and Rock Samples
    #--------------------------------------------------------------------------
    # Find pixels in *_hsv_range and return a single channel binary image.
    navg_select = cv2.inRange(hsv, navg_hsv_range[0], navg_hsv_range[1])
    obst_select = cv2.inRange(hsv, obst_hsv_range[0], obst_hsv_range[1])
    rock_select = cv2.inRange(hsv, rock_hsv_range[0], rock_hsv_range[1])
    # inRange returns array of 0->False, and 255->True, so clip 255 to 1.
    navg_select = np.clip(navg_select, 0, 1)
    obst_select = np.clip(obst_select, 0, 1)
    rock_select = np.clip(rock_select, 0, 1)

    # Trim obstacles to in_view.
    obst_select[np.logical_not(in_view)] = 0

    # Get those pesky bright pixels in the navigable terrain.
    rgb_thresh=(200, 200, 200)
    above_thresh = (img[:,:,0] > rgb_thresh[0]) \
                 & (img[:,:,1] > rgb_thresh[1]) \
                 & (img[:,:,2] > rgb_thresh[2])

    navg_select[above_thresh] = 1

    #--------------------------------------------------------------------------
    # Return the binary images
    #--------------------------------------------------------------------------
    return navg_select, obst_select, rock_select


# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at
    # the center bottom of the image.  
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle) 
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles


# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))
                            
    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result  
    return xpix_rotated, ypix_rotated


def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale): 
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result  
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world


def rock_vector(dist, xpix, ypix, pct_rock=0.01):
    """
    Get vector to rock.
    Return (0.0, 0.0) if no rock is found.
    """
    npix = len(xpix)
    if npix:
        # N_near_rock: number of nearest rock pixels to take
        N_near_rock = np.ceil(pct_rock*npix).astype(np.int)
        I_near_rock = np.argsort(dist)[0:N_near_rock]
        #x_rock = np.mean(xpix[I_near_rock])
        #y_rock = np.mean(ypix[I_near_rock])
        I_mid = I_near_rock[len(I_near_rock)//2]
        x_rock = xpix[I_mid]
        y_rock = ypix[I_mid]
        return x_rock, y_rock
    return 0, 0

################################################################################
# INITIALIZATION
################################################################################
#===============================================================================
# Get HSV ranges.
#===============================================================================
image_dir = '../my_calibration_images'

#-------------------------------------------------------------------------------
# Load Images
#-------------------------------------------------------------------------------
# Load example rocks that have been extracted for calibrating rock_hsv_range.
rock_extracted_img = []
for i in range(3):
    file_PATH = image_dir+'/example_rock'+str(i+1)+'_extracted.jpg'
    rock_extracted_img.append(mpimg.imread(file_PATH))
 
# Navigable
navg_extracted_img = []
for i in range(2):
    file_PATH = image_dir+'/example_navigable'+str(i+1)+'_extracted.jpg'
    navg_extracted_img.append(mpimg.imread(file_PATH))

# Obstacle
obst_extracted_img = mpimg.imread(image_dir+'/example_obstacle_extracted.jpg')

#-------------------------------------------------------------------------------
# Compute HSV ranges
#-------------------------------------------------------------------------------
rock_hsv_range = get_hsv_range(rock_extracted_img, 0.75)
navg_hsv_range = get_hsv_range(navg_extracted_img, 2.75)
obst_hsv_range = get_hsv_range(obst_extracted_img, 3.00)

#===============================================================================
# Set Reference image for resolution 
#===============================================================================
# !!!Needs modified if Roversim resolution changes!!!

refImg = rock_extracted_img[0]

#===============================================================================
# 1) Define source and destination points for perspective transform
#===============================================================================

# Define calibration box in source (actual) and destination (desired)
# coordinates.
# These source and destination points are defined to warp the image
# to a grid where each 10x10 pixel square represents 1 square meter.
# The destination box will be 2*dst_size on each side.
dst_size = 5 
# Set a bottom offset to account for the fact that the bottom of the image 
# is not the position of the rover but a bit in front of it
# this is just a rough guess, feel free to change it!
bottom_offset = 7 
source = np.float32([[22, 139], [295, 138],[199, 97], [119, 97]])
destination = np.float32([
   [refImg.shape[1]/2 - dst_size, refImg.shape[0] - bottom_offset],
   [refImg.shape[1]/2 + dst_size, refImg.shape[0] - bottom_offset],
   [refImg.shape[1]/2 + dst_size, refImg.shape[0] - 2*dst_size - bottom_offset],
   [refImg.shape[1]/2 - dst_size, refImg.shape[0] - 2*dst_size - bottom_offset],
    ])

#===============================================================================
# Get pixels within rover view after perspect_transform.
#===============================================================================
in_view = within_view(refImg, source, destination)

# END INITIALIZATION ###########################################################


################################################################################
# MAIN PERCEPTION FUNCTION 
################################################################################
# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    
    # TODO: 
    # NOTE: camera image is coming to you in Rover.img

    #===========================================================================
    # 1) Define source and destination points for perspective transform
    #===========================================================================

    # Done in "INITIALIZATION" above.

    #===========================================================================
    # 2) Apply perspective transform
    #===========================================================================

    warped = perspect_transform(Rover.img, source, destination)

    #===========================================================================
    # 3) Apply color threshold to identify navigable/obstacles/rock terrain 
    #===========================================================================

    thresh = {}
    thresh['navg'],thresh['obst'],thresh['rock'] = \
            color_thresh_hsv(warped, in_view, \
                             rock_hsv_range, navg_hsv_range, obst_hsv_range)

    #===========================================================================
    # 4) Update Rover.vision_image
    #===========================================================================
        # (this will be displayed on left side of screen)
        # Example: Rover.vision_image[:,:,0] = \
        #                       obstacle color-thresholded binary image
        #          Rover.vision_image[:,:,1] = \
        #                       rock_sample color-thresholded binary image
        #          Rover.vision_image[:,:,2] = \
        #                       navigable terrain color-thresholded binary image
    for i, I in enumerate(('obst', 'rock', 'navg')):
        Rover.vision_image[:,:,i] = thresh[I] * 255

    #===========================================================================
    # 5) Convert map image pixel values to rover-centric coords
    #===========================================================================

    xpix, ypix = {}, {}
    for I in thresh.keys():
        xpix[I], ypix[I] = rover_coords(thresh[I])

    Rover.xpix, Rover.ypix = xpix, ypix

    #===========================================================================
    # 6) Convert rover-centric pixel positions to polar coordinates
    #===========================================================================
        # Update Rover pixel distances and angles
        # Rover.nav_dists = rover_centric_pixel_distances
        # Rover.nav_angles = rover_centric_angles

    dist, angles = {}, {}
    for I in ('navg','obst','rock',):
        dist[I], angles[I] = to_polar_coords(xpix[I], ypix[I])

    Rover.nav_dists  = dist  ['navg']
    Rover.nav_angles = angles['navg']

    # Filter out far away measurements since they are more erroneous
    Rover.scale = float(dst_size*2)
    for I in ('navg', 'obst', 'rock'):
        near_pixels = dist[I] < (Rover.view_limit * Rover.scale)
        xpix[I], ypix[I] = xpix[I][near_pixels], ypix[I][near_pixels]

    # Reset the rock dist and angles after distance filter
    dist['rock'], angles['rock'] = to_polar_coords(xpix['rock'], ypix['rock'])

    # Get the coordinates of the rock
    x_rock, y_rock = rock_vector(dist['rock'], xpix['rock'], ypix['rock'])
    Rover.x_rock, Rover.y_rock = x_rock, y_rock
    Rover.dist_rock, Rover.angle_rock = to_polar_coords(x_rock, y_rock)

    #===========================================================================
    # 7) Convert rover-centric pixel values to world coordinates
    #===========================================================================

    worldsize = Rover.worldmap.shape[0]
    x_pix_world, y_pix_world = {}, {}
    for I in ('navg', 'obst'):
        x_pix_world[I], y_pix_world[I] = \
            pix_to_world(xpix[I], ypix[I],
                         Rover.pos[0], Rover.pos[1],
                         Rover.yaw,
                         worldsize, Rover.scale)

    # Get the world coordinates of the rock
    if len(xpix['rock']): # If rock was visible:
        x_pix_world['rock'], y_pix_world['rock'] = \
            pix_to_world(x_rock, y_rock,
                         Rover.pos[0], Rover.pos[1],
                         Rover.yaw,
                         worldsize, Rover.scale)
        Rover.x_rock_world = x_pix_world['rock']
        Rover.y_rock_world = y_pix_world['rock']
    else: # No rock was visible:
        x_pix_world['rock'], y_pix_world['rock'] = np.int64([]), np.int64([])

    #===========================================================================
    # 8) Update Rover worldmap (to be displayed on right side of screen)
    #===========================================================================
        # Example: Rover.worldmap[obstacle_y_world, obstacle_x_world, 0] += 1
        #          Rover.worldmap[rock_y_world, rock_x_world, 1] += 1
        #          Rover.worldmap[navigable_y_world, navigable_x_world, 2] += 1
 
    tol_p = 0.5       # degrees 
    tol_m = 360-tol_p # degrees 
    #weight = {'obst':1, 'rock':75, 'navg':0.4}
    weight = {'obst':1, 'rock':75, 'navg':0.6}#0.3}

    # Update worldmap score
    if Rover.is_level():
        for i, I in enumerate(('obst', 'rock', 'navg')):
            Rover.worldmap_score[y_pix_world[I], x_pix_world[I], i] += weight[I]

    # Cap obstacle scores to keep them < cap above navg 
    #---------------------------------------------------------------------------
    cap = 75
    obst_over_cap = \
         (Rover.worldmap_score[:,:,0] - Rover.worldmap_score[:,:,2]) > cap
    capped_worldmap_score = np.array(Rover.worldmap_score)
    capped_worldmap_score[:,:,0] = Rover.worldmap_score[:,:,2] + cap
    Rover.worldmap_score[obst_over_cap] = capped_worldmap_score[obst_over_cap]

    # Determine highest score
    #---------------------------------------------------------------------------
    # Mask to get non-zero worldmap pixel scores
    non_0 = np.bool(False)
    for i in range(Rover.worldmap_score.shape[2]):
        non_0 = np.logical_or(non_0, Rover.worldmap_score[:,:,i] > 0.0)
    # Highest score for a non-zero pixel (0 obstacle, 1 rock, or 2 navigable)
    imax = np.argmax(Rover.worldmap_score[non_0], axis=1)


    # Update worldmap
    #---------------------------------------------------------------------------
    # World map colors: obstacle         rock         navigable
    terrain_colors = [[255,  0,  0], [  0,255,  0], [  0,  0,255]]

    # reshape imax into a column vector and choose corresponding terrain color
    Rover.worldmap[non_0] = np.choose(imax.reshape(-1,1), terrain_colors)

    return Rover

# END OF FILE ##################################################################
