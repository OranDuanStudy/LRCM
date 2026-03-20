import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
#import IPython
import os

def save_fig(fig_id, tight_layout=True):
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_id + '.png', format='png', dpi=300)
    
    
def draw_stickfigure(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    if data is None:
        df = mocap_track.values
    else:
        df = data
        
    for joint in joints_to_draw:
        ax.scatter(x=df['%s_Xposition'%joint][frame], 
                   y=df['%s_Yposition'%joint][frame],  
                   alpha=0.6, c='b', marker='o')

        parent_x = df['%s_Xposition'%joint][frame]
        parent_y = df['%s_Yposition'%joint][frame]
        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]
            child_y = df['%s_Yposition'%c][frame]
            ax.plot([parent_x, child_x], [parent_y, child_y], 'k-', lw=2)
            
        if draw_names:
            ax.annotate(joint, 
                    (df['%s_Xposition'%joint][frame] + 0.1, 
                     df['%s_Yposition'%joint][frame] + 0.1))

    return ax

def draw_stickfigure3d(mocap_track, frame, data=None, joints=None, draw_names=False, ax=None, figsize=(8,8)):
    from mpl_toolkits.mplot3d import Axes3D
    
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d') 
    
    if joints is None:
        joints_to_draw = mocap_track.skeleton.keys()
    else:
        joints_to_draw = joints
    
    if data is None:
        df = mocap_track.values
    else:
        df = data
        
    for joint in joints_to_draw:
        parent_x = df['%s_Xposition'%joint][frame]
        parent_y = df['%s_Zposition'%joint][frame]
        parent_z = df['%s_Yposition'%joint][frame]
        # ^ In mocaps, Y is the up-right axis 

        ax.scatter(xs=parent_x, 
                   ys=parent_y,  
                   zs=parent_z,  
                   alpha=0.6, c='b', marker='o')

        
        children_to_draw = [c for c in mocap_track.skeleton[joint]['children'] if c in joints_to_draw]
        
        for c in children_to_draw:
            child_x = df['%s_Xposition'%c][frame]
            child_y = df['%s_Zposition'%c][frame]
            child_z = df['%s_Yposition'%c][frame]
            # ^ In mocaps, Y is the up-right axis

            ax.plot([parent_x, child_x], [parent_y, child_y], [parent_z, child_z], 'k-', lw=2, c='black')
            
        if draw_names:
            ax.text(x=parent_x + 0.1, 
                    y=parent_y + 0.1,
                    z=parent_z + 0.1,
                    s=joint,
                    color='rgba(0,0,0,0.9')

    return ax


def sketch_move(mocap_track, data=None, ax=None, figsize=(16,8)):
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    if data is None:
        data = mocap_track.values

    for frame in range(0, data.shape[0], 4):
#         draw_stickfigure(mocap_track, f, data=data, ax=ax)
        
        for joint in mocap_track.skeleton.keys():
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]
            
            parent_x = data['%s_Xposition'%joint][frame]
            parent_y = data['%s_Yposition'%joint][frame]
            
            frame_alpha = frame/data.shape[0]
            
            for c in children_to_draw:
                child_x = data['%s_Xposition'%c][frame]
                child_y = data['%s_Yposition'%c][frame]
                
                ax.plot([parent_x, child_x], [parent_y, child_y], '-', lw=1, color='gray', alpha=frame_alpha)


def render_mp4(mocap_track, filename, data=None, ax=None, axis_scale=50, elev=45, azim=45, track_character=False, smooth_factor=0.97):
    # Store floor grid data for translation in tracking mode
    floor_data = {}  # Store original floor data
    wframe = None

    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_axis_off()

        ax.view_init(elev=elev, azim=azim)

        xs = np.linspace(-200, 200, 50)
        ys = np.linspace(-200, 200, 50)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros(X.shape)

        # Store original floor data
        floor_data['xs'] = xs
        floor_data['ys'] = ys
        floor_data['X'] = X.copy()
        floor_data['Y'] = Y.copy()
        floor_data['Z'] = Z.copy()

        wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='grey',lw=0.2)

        # fig = plt.figure(figsize=figsize)
        # ax = fig.add_subplot(111)

    if data is None:
        data = mocap_track.values

    fps=int(np.round(1/mocap_track.framerate))
    lines=[]
    lines.append([plt.plot([0,0], [0,0], [0,0], color='red',
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()])[0] for _ in range(len(mocap_track.skeleton.keys()))])

    # Smoothed variables for character tracking
    smoothed_center_x = None
    smoothed_center_y = None
    smoothed_center_z = None

    # Non-tracking mode: calculate bounding box of character motion
    fixed_xlim = None
    fixed_ylim = None
    fixed_zlim = None

    if not track_character:
        # Calculate position range of all joints across all frames
        x_coords = []
        y_coords = []
        z_coords = []

        for joint in mocap_track.skeleton.keys():
            x_coords.extend(data['%s_Xposition'%joint].values)
            y_coords.extend(data['%s_Yposition'%joint].values)
            z_coords.extend(data['%s_Zposition'%joint].values)

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        z_min, z_max = min(z_coords), max(z_coords)

        # Calculate center point and range
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        z_center = (z_min + z_max) / 2

        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min

        # Use larger of axis_scale or actual motion range, add margin
        margin = 1.2  # 20% margin
        scale_x = max(x_range / 2, axis_scale) * margin
        scale_y = max(y_range / 2, axis_scale) * margin
        scale_z = max(z_range / 2, axis_scale) * margin

        fixed_xlim = (x_center - scale_x, x_center + scale_x)
        fixed_ylim = (z_center - scale_z, z_center + scale_z)
        fixed_zlim = (max(0, y_center - scale_y), y_center + scale_y)

        # Set fixed limits during initialization
        ax.set_xlim3d(fixed_xlim[0], fixed_xlim[1])
        ax.set_ylim3d(fixed_ylim[0], fixed_ylim[1])
        ax.set_zlim3d(fixed_zlim[0], fixed_zlim[1])
    else:
        # Tracking mode: set fixed axis range, character stays centered via translation
        # This keeps the character in view without moving the camera
        margin = 1.3  # Add some margin
        ax.set_xlim3d(-axis_scale * margin, axis_scale * margin)
        ax.set_ylim3d(-axis_scale * margin, axis_scale * margin)
        ax.set_zlim3d(max(0, -axis_scale/2 * margin), axis_scale * margin)

    def get_center_position(frame):
        """Get center position (using hips or spine as reference)"""
        center_joints = ['hips', 'spine', 'pelvis', 'torso']
        for joint in center_joints:
            if joint in mocap_track.skeleton:
                x = data['%s_Xposition'%joint].iloc[frame]
                y = data['%s_Yposition'%joint].iloc[frame]
                z = data['%s_Zposition'%joint].iloc[frame]
                return x, y, z

        # If not found, use average position of all joints
        joint_names = list(mocap_track.skeleton.keys())
        x_coords = [data['%s_Xposition'%j].iloc[frame] for j in joint_names]
        y_coords = [data['%s_Yposition'%j].iloc[frame] for j in joint_names]
        z_coords = [data['%s_Zposition'%j].iloc[frame] for j in joint_names]
        return np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)

    def animate(frame):
        nonlocal smoothed_center_x, smoothed_center_y, smoothed_center_z, wframe

        # If tracking mode is enabled
        if track_character:
            center_x, center_y, center_z = get_center_position(frame)

            # Apply smoothing
            if smoothed_center_x is None:
                smoothed_center_x = center_x
                smoothed_center_y = center_y
                smoothed_center_z = center_z
            else:
                smoothed_center_x = smooth_factor * smoothed_center_x + (1 - smooth_factor) * center_x
                smoothed_center_y = smooth_factor * smoothed_center_y + (1 - smooth_factor) * center_y
                smoothed_center_z = smooth_factor * smoothed_center_z + (1 - smooth_factor) * center_z

            # Update floor grid position (remove old, draw new)
            if wframe is not None and floor_data:
                wframe.remove()
                # Translate floor grid to maintain relative position between character and floor
                X_shifted = floor_data['X'] - smoothed_center_x
                Y_shifted = floor_data['Y'] - smoothed_center_z
                Z_shifted = floor_data['Z'] - smoothed_center_y
                wframe = ax.plot_wireframe(X_shifted, Y_shifted, Z_shifted, rstride=2, cstride=2, color='grey', lw=0.2)

        changed = []
        j=0
        for joint in mocap_track.skeleton.keys():
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]

            parent_x = data['%s_Xposition'%joint].iloc[frame]
            parent_y = data['%s_Yposition'%joint].iloc[frame]
            parent_z = data['%s_Zposition'%joint].iloc[frame]

            # Tracking mode: subtract smoothed center offset to keep character centered
            if track_character and smoothed_center_x is not None:
                parent_x -= smoothed_center_x
                parent_y -= smoothed_center_y
                parent_z -= smoothed_center_z

            #frame_alpha = frame/data.shape[0]

            for c in children_to_draw:

                child_x = data['%s_Xposition'%c].iloc[frame]
                child_y = data['%s_Yposition'%c].iloc[frame]
                child_z = data['%s_Zposition'%c].iloc[frame]

                # Tracking mode: subtract smoothed center offset
                if track_character and smoothed_center_x is not None:
                    child_x -= smoothed_center_x
                    child_y -= smoothed_center_y
                    child_z -= smoothed_center_z

                lines[0][j].set_data(np.array([[child_x, parent_x],[-child_z,-parent_z]]))
                lines[0][j].set_3d_properties(np.array([ child_y,parent_y]))

            changed += lines
            j+=1

        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(data.shape[0]), interval=1000/fps)

    if filename != None:
        ani.save(filename, fps=fps, bitrate=13934)
        ani.event_source.stop()
        del ani
        plt.close()    
    try:
        plt.show()
        plt.save()
    except AttributeError as e:
        pass

def viz_cnn_filter(feature_to_viz, mocap_track, data, gap=25):
    fig = plt.figure(figsize=(16,4))
    ax = plt.subplot2grid((1,8),(0,0))
    ax.imshow(feature_to_viz.T, aspect='auto', interpolation='nearest')
    
    ax = plt.subplot2grid((1,8),(0,1), colspan=7)
    for frame in range(feature_to_viz.shape[0]):
        frame_alpha = 0.2#frame/data.shape[0] * 2 + 0.2

        for joint_i, joint in enumerate(mocap_track.skeleton.keys()):
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]

            parent_x = data['%s_Xposition'%joint][frame] + frame * gap
            parent_y = data['%s_Yposition'%joint][frame] 

            ax.scatter(x=parent_x, 
                       y=parent_y,  
                       alpha=0.6,
                       cmap='RdBu',
                       c=feature_to_viz[frame][joint_i] * 10000,
                       marker='o',
                       s = abs(feature_to_viz[frame][joint_i] * 10000))
            plt.axis('off')
            for c in children_to_draw:
                child_x = data['%s_Xposition'%c][frame] + frame * gap
                child_y = data['%s_Yposition'%c][frame] 

                ax.plot([parent_x, child_x], [parent_y, child_y], '-', lw=1, color='gray', alpha=frame_alpha)

                   
def print_skel(X):
    stack = [X.root_name]
    tab=0
    while stack:
        joint = stack.pop()
        tab = len(stack)
        print('%s- %s (%s)'%('| '*tab, joint, X.skeleton[joint]['parent']))
        for c in X.skeleton[joint]['children']:
            stack.append(c)


# def nb_play_mocap_fromurl(mocap, mf, frame_time=1/30, scale=1, base_url='http://titan:8385'):
    # if mf == 'bvh':
        # bw = BVHWriter()
        # with open('test.bvh', 'w') as ofile:
            # bw.write(mocap, ofile)
        
        # filepath = '../notebooks/test.bvh'
    # elif mf == 'pos':
        # c = list(mocap.values.columns)

        # for cc in c:
            # if 'rotation' in cc:
                # c.remove(cc)
        # mocap.values.to_csv('test.csv', index=False, columns=c)
        
        # filepath = '../notebooks/test.csv'
    # else:
        # return
    
    # url = '%s/mocapplayer/player.html?data_url=%s&scale=%f&cz=200&order=xzyi&frame_time=%f'%(base_url, filepath, scale, frame_time)
    # iframe = '<iframe src=' + url + ' width="100%" height=500></iframe>'
    # link = '<a href=%s target="_blank">New Window</a>'%url
    # return IPython.display.HTML(iframe+link)

'''
def nb_play_mocap(mocap, mf, meta=None, frame_time=1/30, scale=1, camera_z=500, base_url=None):
    data_template = 'var dataBuffer = `$$DATA$$`;'
    data_template += 'var metadata = $$META$$;'
    data_template += 'start(dataBuffer, metadata, $$CZ$$, $$SCALE$$, $$FRAMETIME$$);'
    dir_path = os.path.dirname(os.path.realpath(__file__))


    if base_url is None:
      base_url = os.path.join(dir_path, 'mocapplayer/playBuffer.html')

    print(dir_path)

    if mf == 'bvh':
      pass
    elif mf == 'pos':
      cols = list(mocap.values.columns)
      for c in cols:
        if 'rotation' in c:
            cols.remove(c)
    
      data_csv = mocap.values.to_csv(index=False, columns=cols)

      if meta is not None:
        lines = [','.join(item) for item in meta.astype('str')]
        meta_csv = '[' + ','.join('[%s]'%l for l in lines) +']'            
      else:
        meta_csv = '[]'
    
      data_assigned = data_template.replace('$$DATA$$', data_csv)
      data_assigned = data_assigned.replace('$$META$$', meta_csv)
      data_assigned = data_assigned.replace('$$CZ$$', str(camera_z))
      data_assigned = data_assigned.replace('$$SCALE$$', str(scale))
      data_assigned = data_assigned.replace('$$FRAMETIME$$', str(frame_time))

    else:
      return



    with open(os.path.join(dir_path, 'mocapplayer/data.js'), 'w') as oFile:
      oFile.write(data_assigned)

    url = '%s?&cz=200&order=xzyi&frame_time=%f&scale=%f'%(base_url, frame_time, scale)
    iframe = '<iframe frameborder="0" src=' + url + ' width="100%" height=500></iframe>'
    link = '<a href=%s target="_blank">New Window</a>'%url
    return IPython.display.HTML(iframe+link)
'''


def render_mp4_second_person(mocap_track, filename, data=None, ax=None, axis_scale=40, distance=120, height_offset=40, smooth_factor=0.9):
    """
    Second-person view rendering - Camera positioned in front of character, tracking center torso.

    Parameters:
    - mocap_track: Motion capture data
    - filename: Output file name
    - data: Optional data source
    - ax: Optional matplotlib axes
    - axis_scale: Axis range (default 40, moderate view)
    - distance: Camera distance from character (default 120, suitable distance)
    - height_offset: Camera height offset (default 40)
    - smooth_factor: Smoothing factor (0-1, higher is smoother, default 0.9)
    """
    if ax is None:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(True)
        ax.set_axis_off()

        # Create ground grid
        xs = np.linspace(-axis_scale, axis_scale, 20)
        ys = np.linspace(-axis_scale, axis_scale, 20)
        X, Y = np.meshgrid(xs, ys)
        Z = np.zeros(X.shape)
        wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2, color='lightgray', lw=0.5, alpha=0.3)

    if data is None:
        data = mocap_track.values

    fps = int(np.round(1/mocap_track.framerate))

    # Create skeleton lines
    lines = []
    lines.append([plt.plot([0,0], [0,0], [0,0], color='red',
        lw=4, path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()])[0]
        for _ in range(len(mocap_track.skeleton.keys()))])

    # Smoothing variables storage
    smoothed_camera_x = None
    smoothed_camera_y = None
    smoothed_camera_z = None

    def get_center_torso_position(frame):
        """Get center torso position (using hips or spine as reference)"""
        # Try to find hips or spine joint
        center_joints = ['hips', 'spine', 'pelvis', 'torso']
        for joint in center_joints:
            if joint in mocap_track.skeleton:
                x = data['%s_Xposition'%joint].iloc[frame]
                y = data['%s_Yposition'%joint].iloc[frame]
                z = data['%s_Zposition'%joint].iloc[frame]
                return x, y, z

        # If not found, use average position of all joints
        joint_names = list(mocap_track.skeleton.keys())
        x_coords = [data['%s_Xposition'%j].iloc[frame] for j in joint_names]
        y_coords = [data['%s_Yposition'%j].iloc[frame] for j in joint_names]
        z_coords = [data['%s_Zposition'%j].iloc[frame] for j in joint_names]
        return np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)

    def animate(frame):
        nonlocal smoothed_camera_x, smoothed_camera_y, smoothed_camera_z

        # Get center torso position
        center_x, center_y, center_z = get_center_torso_position(frame)

        # Set camera position (in front of character)
        target_camera_x = center_x
        target_camera_y = center_y + height_offset
        target_camera_z = center_z + distance

        # Apply smoothing
        if smoothed_camera_x is None:
            # First frame: use target values directly
            smoothed_camera_x = target_camera_x
            smoothed_camera_y = target_camera_y
            smoothed_camera_z = target_camera_z
        else:
            # Apply exponential moving average for smoothing
            smoothed_camera_x = smooth_factor * smoothed_camera_x + (1 - smooth_factor) * target_camera_x
            smoothed_camera_y = smooth_factor * smoothed_camera_y + (1 - smooth_factor) * target_camera_y
            smoothed_camera_z = smooth_factor * smoothed_camera_z + (1 - smooth_factor) * target_camera_z

        # Set camera to face center (using smoothed position)
        ax.set_xlim3d(smoothed_camera_x - axis_scale, smoothed_camera_x + axis_scale)
        ax.set_ylim3d(smoothed_camera_z - axis_scale, smoothed_camera_z + axis_scale)
        ax.set_zlim3d(max(0, smoothed_camera_y - axis_scale/2), smoothed_camera_y + axis_scale/2)

        # Update camera view - fix flat appearance: add elevation
        ax.view_init(elev=15, azim=90)  # Slight downward angle for better 3D effect

        changed = []
        j = 0

        for joint in mocap_track.skeleton.keys():
            children_to_draw = [c for c in mocap_track.skeleton[joint]['children']]

            parent_x = data['%s_Xposition'%joint].iloc[frame]
            parent_y = data['%s_Yposition'%joint].iloc[frame]
            parent_z = data['%s_Zposition'%joint].iloc[frame]

            for c in children_to_draw:
                child_x = data['%s_Xposition'%c].iloc[frame]
                child_y = data['%s_Yposition'%c].iloc[frame]
                child_z = data['%s_Zposition'%c].iloc[frame]

                # Update line data
                lines[0][j].set_data(np.array([[child_x, parent_x], [child_z, parent_z]]))
                lines[0][j].set_3d_properties(np.array([child_y, parent_y]))

            changed += lines[0][j:j+1]
            j += 1

        return changed

    plt.tight_layout()

    ani = animation.FuncAnimation(fig, animate, np.arange(data.shape[0]),
                               interval=1000/fps, blit=True)

    if filename != None:
        try:
            ani.save(filename, fps=fps, bitrate=13934, writer='ffmpeg')
        except:
            # If ffmpeg is unavailable, use pillow to save gif
            gif_filename = filename.replace('.mp4', '.gif')
            ani.save(gif_filename, fps=fps, writer='pillow')
        ani.event_source.stop()
        del ani
        plt.close()
    try:
        plt.show()
    except AttributeError:
        pass