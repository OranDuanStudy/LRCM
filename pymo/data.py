import numpy as np

class Joint():
    def __init__(self, name, parent=None, children=None):
        self.name = name
        self.parent = parent
        self.children = children

"""
Summary
    `Joint` class:
    - Represents a joint in the skeleton hierarchy
    - Attributes:
        - `name`: Name of the joint
        - `parent`: Parent joint (if any)
        - `children`: List of child joints
 """
class MocapData():
    def __init__(self):
        self.skeleton = {}
        self.values = None
        self.channel_names = []
        self.framerate = 0.0
        self.root_name = ''
        self.take_name = ''

    def traverse(self, j=None):
        stack = [self.root_name]
        while stack:
            joint = stack.pop()
            yield joint
            for c in self.skeleton[joint]['children']:
                stack.append(c)

    def clone(self):
        import copy
        new_data = MocapData()
        new_data.skeleton = copy.deepcopy(self.skeleton)
        new_data.values = copy.deepcopy(self.values)
        new_data.channel_names = copy.deepcopy(self.channel_names)
        new_data.root_name = copy.deepcopy(self.root_name)
        new_data.framerate = copy.deepcopy(self.framerate)
        if hasattr(self,'take_name'):
            new_data.take_name = copy.deepcopy(self.take_name)
        return new_data

    def get_all_channels(self):
        '''Returns all of the channels parsed from the file as a 2D numpy array'''

        frames = [f[1] for f in self.values]
        return np.asarray([[channel[2] for channel in frame] for frame in frames])

    def get_skeleton_tree(self):
        tree = []
        root_key =  [j for j in self.skeleton if self.skeleton[j]['parent']==None][0]

        root_joint = Joint(root_key)

    def get_empty_channels(self):
        #TODO
        pass

    def get_constant_channels(self):
        #TODO
        pass

        """
    `MocapData` class:
    - Represents motion capture data
    - Attributes:
        - `skeleton`: Dictionary representing skeleton hierarchy. Each joint name maps to a dict containing:
            - `parent`: Parent joint name
            - `children`: List of child joint names
        - `values`: List of frames, each frame is a list of channels (e.g., joint rotations)
        - `channel_names`: List of channel names (e.g., "Xrotation", "Yrotation", "Zrotation")
        - `framerate`: Frame rate of the data
        - `root_name`: Name of the root joint
        - `take_name`: Name of the motion capture sample (optional)

    `traverse` method performs depth-first traversal of skeleton hierarchy
    `clone` method creates a deep copy of the MocapData object
    `get_all_channels` method returns all channels as a 2D numpy array
    `get_skeleton_tree` method builds a tree representation of the skeleton hierarchy
    `get_empty_channels` and `get_constant_channels` methods are placeholders for future implementation
        """
