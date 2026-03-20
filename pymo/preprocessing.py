'''
Preprocessing Tranformers Based on sci-kit's API
Created on June 12, 2017
'''
import copy
import pandas as pd
import numpy as np
import transforms3d as t3d
import scipy.ndimage.filters as filters
from scipy.spatial.transform import Rotation as R

from scipy import signal, interpolate
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from pymo.rotation_tools import Rotation, euler2expmap, euler2expmap2, expmap2euler, euler_reorder, unroll, euler2vectors, vectors2euler
from pymo.Quaternions import Quaternions
from pymo.Pivots import Pivots

class MocapParameterizer(BaseEstimator, TransformerMixin):
    def __init__(self, param_type = 'euler', ref_pose=None):
        '''
        Initialization function

        Args:
        param_type: Parameter type, options are {'euler', 'quat', 'expmap', 'position', 'expmap2pos'}, default is 'euler'.
        ref_pose: Reference pose, can be any pose representation. If not None, will be converted to quaternion form for storage. Default is None.

        Returns:
        None
        '''
        self.param_type = param_type  # Store parameter type
        if (ref_pose is not None):  # If reference pose is provided
            self.ref_pose = self._to_quat(ref_pose)[0]  # Convert reference pose to quaternion and store
        else:
            self.ref_pose = None  # When reference pose is not provided, default to None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Transform input data X according to the specified parameter type.

        Args:
        - X: Input data, matrix or vector to be transformed.
        - y: Optional parameter, currently unused.

        Returns:
        - Transformed data matrix or vector, depending on parameter type.

        Based on param_type, input data can be converted to Euler angles, rotation vectors, quaternions, position vectors, etc.
        """
        if self.param_type == 'euler':
            return X
        elif self.param_type == 'expmap':
            if self.ref_pose is None:
                return self._to_expmap(X)  # When no reference pose, directly convert Euler angles to exponential map
            else:
                return self._to_expmap2(X)  # When reference pose exists, perform exponential map conversion relative to reference pose
        elif self.param_type == 'vectors':
            return self._euler_to_vectors(X)  # Convert Euler angles to rotation vectors
        elif self.param_type == 'quat':
            return self._to_quat(X)  # Convert Euler angles to quaternions
        elif self.param_type == 'position':
            return self._to_pos(X)  # Extract position information only
        elif self.param_type == 'expmap2pos':
            return self._expmap_to_pos(X)  # Convert exponential map to position information
        else:
            raise 'param types: euler, quat, expmap, position, expmap2pos'  # Raise exception for unsupported parameter types

#        return X
    
    def inverse_transform(self, X, copy=None):
        """
        Perform inverse transformation on given parameter X. Transformation type depends on instance's param_type attribute.

        Args:
        - X: Dataset to be inversely transformed, its specific type and structure depend on param_type.
        - copy: (Optional) Whether to copy X before operation, defaults to None, meaning decide whether to copy as needed.

        Returns:
        - X after inverse transformation, its type and structure depend on param_type and specific implementation.
        """
        if self.param_type == 'euler':
            # If parameter type is Euler angles, return X directly without any transformation
            return X
        elif self.param_type == 'expmap':
            # If parameter type is exponential map, choose different inverse transformation method based on whether ref_pose exists
            if self.ref_pose is None:
                return self._expmap_to_euler(X)
            else:
                return self._expmap_to_euler2(X)
        elif self.param_type == 'vectors':
            # If parameter type is vectors, inverse transform to Euler angles
            return self._vectors_to_euler(X)
        elif self.param_type == 'quat':
            # If parameter type is quaternions, inverse transform to Euler angles
            return self._quat_to_euler(X)
        elif self.param_type == 'position':
            # If parameter type is position, this type of inverse transformation is not currently supported, just print a message and return original X
            print('positions 2 eulers is not supported')
            return X
        else:
            # If param_type is not a supported type, raise exception
            raise 'param types: euler, quat, expmap, position'

    def _to_quat(self, X):
        '''
        Convert joint rotations from Euler angles to quaternions.

        Args:
        X - A data structure containing multiple trajectories (each trajectory contains Euler angle rotation information for joints)

        Returns:
        Q - List of trajectories with quaternion rotation information after conversion
        '''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store exponential map representation
            quat_df = euler_df.copy()

            # List columns containing rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List joints that are not end sites (i.e., have channels)
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                rot_order = track.skeleton[joint]['order']

                # Get rotation columns belonging to this joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                # Ensure columns are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))

                quat_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)

                # Create corresponding columns in the new DataFrame
                quat_df['%s_qWrotation'%joint] = pd.Series(data=[e[0] for e in quats], index=quat_df.index)
                quat_df['%s_qXrotation'%joint] = pd.Series(data=[e[1] for e in quats], index=quat_df.index)
                quat_df['%s_qYrotation'%joint] = pd.Series(data=[e[2] for e in quats], index=quat_df.index)
                quat_df['%s_qZrotation'%joint] = pd.Series(data=[e[3] for e in quats], index=quat_df.index)

            new_track = track.clone()
            new_track.values = quat_df
            Q.append(new_track)
        return Q
    
    def _quat_to_euler(self, X):
        """
        Convert quaternion rotation channels to Euler angles.

        Args:
        X - A list of trajectory data, where each trajectory contains rotation channels (quaternion representation).

        Returns:
        Q - List of trajectories after conversion, where rotation channels are in Euler angle format.
        """
        Q = []
        for track in X:
            channels = []
            titles = []
            quat_df = track.values

            # Copy original data frame to preserve quaternion representation
            euler_df = quat_df.copy()

            # Determine columns containing rotation channels
            quat_params = [c for c in quat_df.columns if (any(p in c for p in ['qWrotation','qXrotation','qYrotation','qZrotation']) and 'Nub' not in c)]

            # Determine non-end joints, i.e., joints with channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                # Get columns belonging to this joint
                r = quat_df[[c for c in quat_params if joint in c]]

                # Remove quaternion columns from data frame, prepare to replace with Euler angles
                euler_df.drop(['%s_qWrotation'%joint, '%s_qXrotation'%joint, '%s_qYrotation'%joint, '%s_qZrotation'%joint], axis=1, inplace=True)

                # Organize quaternion columns for conversion
                quat = [[f[1]['%s_qWrotation'%joint], f[1]['%s_qXrotation'%joint], f[1]['%s_qYrotation'%joint], f[1]['%s_qZrotation'%joint]] for f in r.iterrows()]
                quats=Quaternions(np.asarray(quat))
                euler_rots = 180/np.pi*quats.euler()
                track.skeleton[joint]['order'] = 'ZYX'
                rot_order = track.skeleton[joint]['order']

                # Create new Euler angle columns in data frame
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            # Create new trajectory with converted data frame as values
            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q
    
    def _to_pos(self, X):
        """
        Convert joint Euler angle rotations to their corresponding positions.

        Args:
        X - List of trajectories containing joint rotations (Euler angle representation) and position information.

        Returns:
        Q - List of converted trajectories, where joint positions are calculated based on their rotations and hierarchy.
        """

        Q = []

        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store final position data
            pos_df = pd.DataFrame(index=euler_df.index)

            # Get column names containing rotation channels
            rot_cols = [c for c in euler_df.columns if ('rotation' in c)]

            # Get column names containing position channels
            pos_cols = [c for c in euler_df.columns if ('position' in c)]

            # Get all non-end joints (i.e., joints with channels)
            joints = (joint for joint in track.skeleton)

            # Initialize joint tree data dictionary for storing intermediate calculation results
            tree_data = {}

            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                rot_order = track.skeleton[joint]['order']

                # Get rotation columns corresponding to current joint
                rc = euler_df[[c for c in rot_cols if joint in c]]

                # Get position columns corresponding to current joint
                pc = euler_df[[c for c in pos_cols if joint in c]]

                # Ensure rotation and position data are organized in xyz order
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi / 180.0 * np.transpose(
                        np.array([
                            track.values['%s_%srotation' % (joint, rot_order[0])],
                            track.values['%s_%srotation' % (joint, rot_order[1])],
                            track.values['%s_%srotation' % (joint, rot_order[2])]
                        ])
                    )

                if pc.shape[1] < 3:
                    pos_values = np.asarray([[0, 0, 0] for f in pc.iterrows()])
                else:
                    pos_values = np.asarray([[
                        f[1]['%s_Xposition' % joint],
                        f[1]['%s_Yposition' % joint],
                        f[1]['%s_Zposition' % joint]
                    ] for f in pc.iterrows()])

                # Convert Euler angles to quaternions
                quats = Quaternions.from_euler(np.asarray(euler_values),
                                               order=rot_order.lower(),
                                               world=False)

                # Initialize joint tree data entries
                tree_data[joint] = [
                    [],  # Store rotation matrices
                    []  # Store calculated positions
                ]

                if track.root_name == joint:
                    tree_data[joint][0] = quats  # rotmats
                    tree_data[joint][1] = pos_values  # Add offset
                else:
                    # For each frame, multiply current joint's rotation matrix with parent's rotation matrix
                    tree_data[joint][0] = tree_data[parent][0] * quats  # np.matmul(rotmats, tree_data[parent][0])

                    # Calculate and store joint position relative to parent
                    k = pos_values + np.asarray(track.skeleton[joint]['offsets'])
                    q = tree_data[parent][0] * k  # np.matmul(k.reshape(k.shape[0], 1, 3), tree_data[parent][0])
                    tree_data[joint][1] = tree_data[parent][1] + q  # q.reshape(k.shape[0], 3) + tree_data[parent][1]

                # Add calculated position data to new DataFrame
                df = pd.DataFrame(
                    data=tree_data[joint][1],
                    index=pos_df.index,
                    columns=['%s_Xposition' % joint, '%s_Yposition' % joint, '%s_Zposition' % joint]
                )
                pos_df = pd.concat((pos_df, df), axis=1)

            # Create new trajectory object using calculated position data as values
            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)

        return Q

    def _expmap2rot(self, expmap):
        """
        Convert exponential map (expmap) to rotation matrix (rotmat).

        Args:
        expmap: A 3D array containing the exponential map representation of rotations.

        Returns:
        rotmats: A 4D array containing the corresponding rotation matrix representation.
        """

        # Calculate the norm of each vector and normalize non-zero norm vectors
        theta = np.linalg.norm(expmap, axis=1, keepdims=True)
        nz = np.nonzero(theta)[0]
        expmap[nz,:] = expmap[nz,:]/theta[nz]

        # Extract x, y, z components separately
        nrows=expmap.shape[0]
        x = expmap[:,0]
        y = expmap[:,1]
        z = expmap[:,2]

        # Calculate sine and cosine values, reshape for easier computation
        s = np.sin(theta*0.5).reshape(nrows)
        c = np.cos(theta*0.5).reshape(nrows)

        # Initialize rotation matrix
        rotmats = np.zeros((nrows, 3, 3))

        # Calculate each term according to rotation matrix formula and fill in the rotation matrix
        rotmats[:,0,0] = 2*(x*x-1)*s*s+1
        rotmats[:,0,1] = 2*x*y*s*s-2*z*c*s
        rotmats[:,0,2] = 2*x*z*s*s+2*y*c*s
        rotmats[:,1,0] = 2*x*y*s*s+2*z*c*s
        rotmats[:,1,1] = 2*(y*y-1)*s*s+1
        rotmats[:,1,2] = 2*y*z*s*s-2*x*c*s
        rotmats[:,2,0] = 2*x*z*s*s-2*y*c*s
        rotmats[:,2,1] =  2*y*z*s*s+2*x*c*s
        rotmats[:,2,2] =  2*(z*z-1)*s*s+1

        return rotmats

    def _expmap_to_pos(self, X):
        '''Converts joints rotations in expmap notation to joint positions'''

        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            pos_df = pd.DataFrame(index=exp_df.index)

            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton)

            tree_data = {}
                        
            for joint in track.traverse():
                parent = track.skeleton[joint]['parent']
                
                if 'Nub' not in joint:
                    r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                    expmap = r.values
                    #expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()]
                else:
                    expmap = np.zeros((exp_df.shape[0], 3))

                # Convert the eulers to rotation matrices
                #rotmats = np.asarray([Rotation(f, 'expmap').rotmat for f in expmap])
                #angs = np.linalg.norm(expmap,axis=1, keepdims=True)
                rotmats = self._expmap2rot(expmap)
                
                tree_data[joint]=[
                                    [], # to store the rotation matrix
                                    []  # to store the calculated position
                                 ] 
                pos_values = np.zeros((exp_df.shape[0], 3))

                if track.root_name == joint:
                    tree_data[joint][0] = rotmats
                    # tree_data[joint][1] = np.add(pos_values, track.skeleton[joint]['offsets'])
                    tree_data[joint][1] = pos_values
                else:
                    # for every frame i, multiply this joint's rotmat to the rotmat of its parent
                    tree_data[joint][0] = np.matmul(rotmats, tree_data[parent][0])

                    # add the position channel to the offset and store it in k, for every frame i
                    k = pos_values + track.skeleton[joint]['offsets']

                    # multiply k to the rotmat of the parent for every frame i
                    q = np.matmul(k.reshape(k.shape[0],1,3), tree_data[parent][0])

                    # add q to the position of the parent, for every frame i
                    tree_data[joint][1] = q.reshape(k.shape[0],3) + tree_data[parent][1]


                # Create the corresponding columns in the new DataFrame
                df = pd.DataFrame(data=tree_data[joint][1], 
                                  index=pos_df.index, 
                                  columns=['%s_Xposition'%joint, '%s_Yposition'%joint, '%s_Zposition'%joint])
                pos_df = pd.concat((pos_df, df), axis=1)

            new_track = track.clone()
            new_track.values = pos_df
            Q.append(new_track)
        return Q

    def _to_expmap(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                #print(joint)
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = np.transpose(np.array([r[r1_col], r[r2_col], r[r3_col]]))
                #exps = [Rotation(f, 'euler', from_deg=True, order=rot_order).to_expmap() for f in euler] # Convert the eulers to exp maps
                exps = unroll(np.array([euler2expmap(f, rot_order, True) for f in euler])) # Convert the exp maps to eulers
                #exps = euler2expmap2(euler, rot_order, True) # Convert the eulers to exp maps

                # Fix warning
                # Create the corresponding columns in the new DataFrame    
                ### exp_df.insert(loc=0, column='%s_gamma'%joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                ### exp_df.insert(loc=0, column='%s_beta'%joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                ### exp_df.insert(loc=0, column='%s_alpha'%joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

                # Prepare data for new columns
                new_data = {f'{joint}_{col}': [e[i] for e in exps] for i, col in enumerate(['alpha', 'beta', 'gamma'])}
                # Create a new DataFrame from the prepared data, ensuring the index matches exp_df's index
                new_columns_df = pd.DataFrame(new_data, index=exp_df.index)
                # Concatenate the new columns to the original DataFrame along the column axis (axis=1)
                exp_df = pd.concat([new_columns_df, exp_df], axis=1)
                
            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q
                
    def _expmap_to_euler(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            euler_df = exp_df.copy()
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                jt_alpha = '%s_alpha'%joint
                jt_beta = '%s_beta'%joint
                jt_gamma = '%s_gamma'%joint
                
                euler_df.drop([jt_alpha, jt_beta, jt_gamma], axis=1, inplace=True)
                expmap = np.transpose(np.array([track.values[jt_alpha], track.values[jt_beta], track.values[jt_gamma]]))
                rot_order = track.skeleton[joint]['order']
                euler_rots = np.array(R.from_rotvec(expmap).as_euler(rot_order, degrees=True))                
                #euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                
                # Create the corresponding columns in the new DataFrame    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q

    def _to_expmap2(self, X):
        '''Converts Euler angles to Exponential Maps'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            exp_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # Copy the root positions into the new DataFrame
            #rxp = '%s_Xposition'%track.root_name
            #ryp = '%s_Yposition'%track.root_name
            #rzp = '%s_Zposition'%track.root_name
            #exp_df[rxp] = pd.Series(data=euler_df[rxp], index=exp_df.index)
            #exp_df[ryp] = pd.Series(data=euler_df[ryp], index=exp_df.index)
            #exp_df[rzp] = pd.Series(data=euler_df[rzp], index=exp_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']

                # Get the rotation columns that belong to this joint
                rc = euler_df[[c for c in rots if joint in c]]

                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                # Make sure the columns are organized in xyz order
                #print("joint:" + str(joint) + "  rot_order:" + str(rot_order))
                if rc.shape[1] < 3:
                    euler_values = np.zeros((euler_df.shape[0], 3))
                    rot_order = "XYZ"
                else:
                    euler_values = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                quats = Quaternions.from_euler(np.asarray(euler_values), order=rot_order.lower(), world=False)
                #exps = [Rotation(f, 'euler', from_deg=True, order=rot_order).to_expmap() for f in euler] # Convert the eulers to exp maps
                #exps = unroll(np.array([euler2expmap(f, rot_order, True) for f in euler])) # Convert the exp maps to eulers
                #exps = euler2expmap2(euler, rot_order, True) # Convert the eulers to exp maps
                # Create the corresponding columns in the new DataFrame
                if (self.ref_pose is not None):
                    q1_col = '%s_qWrotation'%(joint)
                    q2_col = '%s_qXrotation'%(joint)
                    q3_col = '%s_qYrotation'%(joint)
                    q4_col = '%s_qZrotation'%(joint)
                    ref_q = Quaternions(np.asarray([[f[1][q1_col], f[1][q2_col], f[1][q3_col], f[1][q4_col]] for f in self.ref_pose.values.iterrows()]))
                    #print("ref_q:" + str(ref_q.shape))
                    ref_q = ref_q[0,:]
                    quats=(-ref_q)*quats
    
                angles, axis = quats.angle_axis()
                aa = np.where(angles>np.pi)
                angles[aa] = angles[aa]-2*np.pi                
                #exps = unroll(angles[:,None]*axis)
                exps = angles[:,None]*axis
                #print(f"{joint}: {str(exps[0,:])}")

                #exps = np.array([quat2expmap(f) for f in quats])
                exp_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                exp_df.insert(loc=0, column='%s_gamma'%joint, value=pd.Series(data=[e[2] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_beta'%joint, value=pd.Series(data=[e[1] for e in exps], index=exp_df.index))
                exp_df.insert(loc=0, column='%s_alpha'%joint, value=pd.Series(data=[e[0] for e in exps], index=exp_df.index))

            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = exp_df
            Q.append(new_track)

        return Q
        
    def _expmap_to_euler2(self, X):
        Q = []
        for track in X:
            channels = []
            titles = []
            exp_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = exp_df.copy()

            # Copy the root positions into the new DataFrame
            #rxp = '%s_Xposition'%track.root_name
            #ryp = '%s_Yposition'%track.root_name
            #rzp = '%s_Zposition'%track.root_name
            #euler_df[rxp] = pd.Series(data=exp_df[rxp], index=euler_df.index)
            #euler_df[ryp] = pd.Series(data=exp_df[ryp], index=euler_df.index)
            #euler_df[rzp] = pd.Series(data=exp_df[rzp], index=euler_df.index)
            
            # List the columns that contain rotation channels
            exp_params = [c for c in exp_df.columns if ( any(p in c for p in ['alpha', 'beta','gamma']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = exp_df[[c for c in exp_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_alpha'%joint, '%s_beta'%joint, '%s_gamma'%joint], axis=1, inplace=True)
                expmap = [[f[1]['%s_alpha'%joint], f[1]['%s_beta'%joint], f[1]['%s_gamma'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                angs = np.linalg.norm(expmap, axis=1)
                quats=Quaternions.from_angle_axis(angs, expmap/(np.tile(angs[:, None]+1e-10, (1,3))))
                if (self.ref_pose is not None):
                    q1_col = '%s_qWrotation'%(joint)
                    q2_col = '%s_qXrotation'%(joint)
                    q3_col = '%s_qYrotation'%(joint)
                    q4_col = '%s_qZrotation'%(joint)
                    ref_q = Quaternions(np.asarray([[f[1][q1_col], f[1][q2_col], f[1][q3_col], f[1][q4_col]] for f in self.ref_pose.values.iterrows()]))
                    #print("ref_q:" + str(ref_q.shape))
                    ref_q = ref_q[0,:]
                    quats=ref_q*quats
                    
                euler_rots = 180/np.pi*quats.euler()
                track.skeleton[joint]['order'] = 'ZYX'
                rot_order = track.skeleton[joint]['order']
                #euler_rots = [Rotation(f, 'expmap').to_euler(True, rot_order) for f in expmap] # Convert the exp maps to eulers
                #euler_rots = [expmap2euler(f, rot_order, True) for f in expmap] # Convert the exp maps to eulers
                                                  
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q
        
    def _euler_to_vectors(self, X):
        '''Converts Euler angles to Up and Fwd vectors'''

        Q = []
        for track in X:
            channels = []
            titles = []
            euler_df = track.values

            # Create a new DataFrame to store the exponential map rep
            vec_df = euler_df.copy()# pd.DataFrame(index=euler_df.index)

            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                #print(joint)
                r = euler_df[[c for c in rots if joint in c]] # Get the columns that belong to this joint
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])
                
                vec_df.drop([r1_col, r2_col, r3_col], axis=1, inplace=True)
                euler = [[f[1][r1_col], f[1][r2_col], f[1][r3_col]] for f in r.iterrows()]
                vectors = np.array([euler2vectors(f, rot_order, True) for f in euler])

                new_columns = []
                for i in range(6):
                    column_name = f"{joint}_{'xyzUpFwd'[i]}{['','Up','Up','Fwd','Fwd','Fwd'][i]}"
                    new_column = pd.Series([e[i] for e in vectors], index=vec_df.index, name=column_name)
                    new_columns.append(new_column)
                
                # Use pd.concat() to add all new columns to the beginning of vec_df at once
                vec_df = pd.concat(new_columns + [vec_df], axis=1)
                '''
                vec_df.insert(loc=0, column='%s_xUp'%joint, value=pd.Series(data=[e[0] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_yUp'%joint, value=pd.Series(data=[e[1] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_zUp'%joint, value=pd.Series(data=[e[2] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_xFwd'%joint, value=pd.Series(data=[e[3] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_yFwd'%joint, value=pd.Series(data=[e[4] for e in vectors], index=vec_df.index))
                vec_df.insert(loc=0, column='%s_zFwd'%joint, value=pd.Series(data=[e[5] for e in vectors], index=vec_df.index))
                '''
            #print(exp_df.columns)
            new_track = track.clone()
            new_track.values = vec_df
            Q.append(new_track)

        return Q
            
    def _vectors_to_euler(self, X):
        '''Converts Up and Fwd vectors to Euler angles'''
        Q = []
        for track in X:
            channels = []
            titles = []
            vec_df = track.values

            # Create a new DataFrame to store the exponential map rep
            #euler_df = pd.DataFrame(index=exp_df.index)
            euler_df = vec_df.copy()

            # List the columns that contain rotation channels
            vec_params = [c for c in vec_df.columns if ( any(p in c for p in ['xUp', 'yUp','zUp','xFwd', 'yFwd','zFwd']) and 'Nub' not in c)]

            # List the joints that are not end sites, i.e., have channels
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            for joint in joints:
                r = vec_df[[c for c in vec_params if joint in c]] # Get the columns that belong to this joint
                
                euler_df.drop(['%s_xUp'%joint, '%s_yUp'%joint, '%s_zUp'%joint, '%s_xFwd'%joint, '%s_yFwd'%joint, '%s_zFwd'%joint], axis=1, inplace=True)
                vectors = [[f[1]['%s_xUp'%joint], f[1]['%s_yUp'%joint], f[1]['%s_zUp'%joint], f[1]['%s_xFwd'%joint], f[1]['%s_yFwd'%joint], f[1]['%s_zFwd'%joint]] for f in r.iterrows()] # Make sure the columsn are organized in xyz order
                rot_order = track.skeleton[joint]['order']
                euler_rots = [vectors2euler(f, rot_order, True) for f in vectors]
                
                # Create the corresponding columns in the new DataFrame
    
                euler_df['%s_%srotation'%(joint, rot_order[0])] = pd.Series(data=[e[0] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[1])] = pd.Series(data=[e[1] for e in euler_rots], index=euler_df.index)
                euler_df['%s_%srotation'%(joint, rot_order[2])] = pd.Series(data=[e[2] for e in euler_rots], index=euler_df.index)

            new_track = track.clone()
            new_track.values = euler_df
            Q.append(new_track)

        return Q

class Mirror(BaseEstimator, TransformerMixin):
    """
    Data mirroring class for applying mirror transformation to motion capture data.

    Args:
    - axis: Mirror axis, can be "X", "Y", or "Z", default is "X".
    - append: Whether to append mirrored data after original data, default is True.
    """
    def __init__(self, axis="X", append=True):
        """
        Initialize mirroring class instance.
        """
        self.axis = axis
        self.append = append
    
    def fit(self, X, y=None):
        """
        Fitting method for mirror transformation.

        Args:
        - X: Data to be processed.
        - y: (Optional) Target values.

        Returns:
        - self: The mirror transformation object itself.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Method to apply mirror transformation to given data.

        Args:
        - X: Dataset to be mirrored, containing multiple trajectories.
        - y: (Optional) Target values.

        Returns:
        - New dataset after mirror transformation.
        """
        # Initialize list to store transformed trajectories
        Q = []

        # If append is True, copy original trajectories first
        if self.append:
            for track in X:
                Q.append(track)
        
        # Iterate through each trajectory for mirror transformation
        for track in X:
            # Determine mirror axis based on axis parameter, set corresponding sign factors
            channels = []
            titles = []
            
            if self.axis == "X":
                signs = np.array([1,-1,-1])
            elif self.axis == "Y":
                signs = np.array([-1,1,-1])
            elif self.axis == "Z":
                signs = np.array([-1,-1,1])
            
            # Get trajectory Euler angle data
            euler_df = track.values

            # Create a new DataFrame to store mirrored data
            # Create a new DataFrame to store the exponential map rep
            new_df = pd.DataFrame(index=euler_df.index)

            # Mirror root position based on mirror axis
            # Copy the root positions into the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=-signs[0]*euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=-signs[1]*euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=-signs[2]*euler_df[rzp], index=new_df.index)
            
            # Iterate through all rotation channels for mirror transformation
            # List the columns that contain rotation channels
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]
            lft_joints = (joint for joint in track.skeleton if 'Left' in joint and 'Nub' not in joint)
            rgt_joints = (joint for joint in track.skeleton if 'Right' in joint and 'Nub' not in joint)
            
            new_track = track.clone()
            
            # Apply mirror transformation to left and right joints
            for lft_joint in lft_joints:
                rgt_joint = lft_joint.replace('Left', 'Right')
                
                # Create mirrored rotation angle columns in new DataFrame

                new_columns = {
                    '%s_Xrotation' % lft_joint: signs[0] * track.values['%s_Xrotation' % rgt_joint],
                    '%s_Yrotation' % lft_joint: signs[1] * track.values['%s_Yrotation' % rgt_joint],
                    '%s_Zrotation' % lft_joint: signs[2] * track.values['%s_Zrotation' % rgt_joint],
                    '%s_Xrotation' % rgt_joint: signs[0] * track.values['%s_Xrotation' % lft_joint],
                    '%s_Yrotation' % rgt_joint: signs[1] * track.values['%s_Yrotation' % lft_joint],
                    '%s_Zrotation' % rgt_joint: signs[2] * track.values['%s_Zrotation' % lft_joint],
                }
                # Create a temporary DataFrame with these new columns
                temp_df = pd.DataFrame(new_columns, index=new_df.index)
                # Concatenate the new columns with the existing DataFrame
                new_df = pd.concat([new_df, temp_df], axis=1)
                # Optional: Create a copy to de-fragment the DataFrame
                new_df = new_df.copy()
                
                ### new_df['%s_Xrotation'%lft_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%rgt_joint], index=new_df.index)
                ### new_df['%s_Yrotation'%lft_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%rgt_joint], index=new_df.index)
                ### new_df['%s_Zrotation'%lft_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%rgt_joint], index=new_df.index)
                ### 
                ### new_df['%s_Xrotation'%rgt_joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%lft_joint], index=new_df.index)
                ### new_df['%s_Yrotation'%rgt_joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%lft_joint], index=new_df.index)
                ### new_df['%s_Zrotation'%rgt_joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%lft_joint], index=new_df.index)
            
            # Apply mirror transformation to non-left-right trunk joints
            joints = (joint for joint in track.skeleton if 'Nub' not in joint and 'Left' not in joint and 'Right' not in joint)
            for joint in joints:
                # Create mirrored rotation angle columns in new DataFrame
                new_df['%s_Xrotation'%joint] = pd.Series(data=signs[0]*track.values['%s_Xrotation'%joint], index=new_df.index)
                new_df['%s_Yrotation'%joint] = pd.Series(data=signs[1]*track.values['%s_Yrotation'%joint], index=new_df.index)
                new_df['%s_Zrotation'%joint] = pd.Series(data=signs[2]*track.values['%s_Zrotation'%joint], index=new_df.index)

            new_track.values = new_df
            new_track.take_name = track.take_name + "_mirrored"
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        """
        Reverse operation of mirror transformation, i.e., restore original data.

        Args:
        - X: Dataset after mirror transformation.
        - copy: (Optional) Whether to copy data, default is None.
        - start_pos: (Optional) Starting position, default is None.

        Returns:
        - Original dataset.
        """
        return X

class EulerReorder(BaseEstimator, TransformerMixin):
    """
    Transformer for reordering Euler angles.

    Args:
    new_order: list
        New sequence for Euler angle reordering.
    """
    def __init__(self, new_order):
        """
        Initialize the transformer and set the new Euler angle order.
        """
        self.new_order = new_order
        
    def fit(self, X, y=None):
        """
        Fit the transformer. During this process, save the original skeleton structure.

        Args:
        X: list
            List of trajectory data.
        y: None
            Unused, kept for API compliance.

        Returns:
        self: object
            Returns the transformer instance.
        """
        self.orig_skeleton = copy.deepcopy(X[0].skeleton)
        return self
    
    def transform(self, X, y=None):
        """
        Transform to reorder Euler angles.

        Args:
        X: list
            List of trajectory data.
        y: None
            Unused, kept for API compliance.

        Returns:
        list
            List of trajectory data with reordered Euler angles.
        """
        # Initialize result list
        Q = []

        for track in X:
            # Initialize channels and titles lists
            channels = []
            titles = []
            # Get the DataFrame containing Euler angle data
            euler_df = track.values

            # Create a new DataFrame to store the exponential map representation
            #new_df = pd.DataFrame(index=euler_df.index)
            new_df = euler_df.copy()

            # Copy root positions to the new DataFrame
            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name
            new_df[rxp] = pd.Series(data=euler_df[rxp], index=new_df.index)
            new_df[ryp] = pd.Series(data=euler_df[ryp], index=new_df.index)
            new_df[rzp] = pd.Series(data=euler_df[rzp], index=new_df.index)

            # List all rotation channel column names
            rots = [c for c in euler_df.columns if ('rotation' in c and 'Nub' not in c)]

            # List all non-end effector joints
            joints = (joint for joint in track.skeleton if 'Nub' not in joint)

            new_track = track.clone()
            for joint in joints:
                # Get rotation channel data belonging to this joint
                r = euler_df[[c for c in rots if joint in c]] 
                rot_order = track.skeleton[joint]['order']
                r1_col = '%s_%srotation'%(joint, rot_order[0])
                r2_col = '%s_%srotation'%(joint, rot_order[1])
                r3_col = '%s_%srotation'%(joint, rot_order[2])

                # Reorder Euler angles
                euler = np.transpose(np.array([r[r1_col], r[r2_col], r[r3_col]]))
                new_euler = [euler_reorder(f, rot_order, self.new_order, True) for f in euler]

                # Create reordered Euler angle columns in the new DataFrame
                new_df['%s_%srotation'%(joint, self.new_order[0])] = pd.Series(data=[e[0] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[1])] = pd.Series(data=[e[1] for e in new_euler], index=new_df.index)
                new_df['%s_%srotation'%(joint, self.new_order[2])] = pd.Series(data=[e[2] for e in new_euler], index=new_df.index)

                # Update joint rotation order
                new_track.skeleton[joint]['order'] = self.new_order

            # Update trajectory data to the new DataFrame
            new_track.values = new_df
            Q.append(new_track)

        return Q
        
    def inverse_transform(self, X, copy=None, start_pos=None):
        """
        Restore the original Euler angle order.

        Args:
        X: list
            List of trajectory data.
        copy: None
            Unused, kept for API compliance.
        start_pos: None
            Unused, kept for API compliance.

        Returns:
        list
            List of trajectory data with original Euler angle order restored.
        """
        return X

class JointSelector(BaseEstimator, TransformerMixin):
    '''
    Allows for filtering the mocap data to include only the selected joints
    '''
    def __init__(self, joints, include_root=False):
        self.joints = joints
        self.include_root = include_root

    def fit(self, X, y=None):
        selected_joints = []
        selected_channels = []

        if self.include_root:
            selected_joints.append(X[0].root_name)
        
        selected_joints.extend(self.joints)

        for joint_name in selected_joints:
            if joint_name.endswith("_Nub"):
                selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o])
            else:
                selected_channels.extend([o for o in X[0].values.columns if (joint_name + "_") in o and 'Nub' not in o])
        
        self.selected_joints = selected_joints
        self.selected_channels = selected_channels
        self.not_selected = X[0].values.columns.difference(selected_channels)
        self.not_selected_values = {c:X[0].values[c].values[0] for c in self.not_selected}

        self.orig_skeleton = X[0].skeleton
        return self

    def transform(self, X, y=None):
        #print("JointSelector")
        Q = []
        for track in X:
            t2 = track.clone()
            for key in track.skeleton.keys():
                if key not in self.selected_joints:
                    t2.skeleton.pop(key)
            t2.values = track.values[self.selected_channels]

            for key in t2.skeleton.keys():
                to_remove = list(set(t2.skeleton[key]['children']) - set(self.selected_joints))
                [t2.skeleton[key]['children'].remove(c) for c in to_remove]

            Q.append(t2)
      

        return Q
    
    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            t2 = track.clone()
            skeleton = self.orig_skeleton
            for key in track.skeleton.keys():
                skeleton[key]['order']=track.skeleton[key]['order']            
            
            
            t2.skeleton = skeleton
            for d in self.not_selected:
                t2.values[d] = self.not_selected_values[d].copy() # t2.values[d] = self.not_selected_values[d]
            Q.append(t2)
            '''
            t2.skeleton = skeleton
        
            # Create a new DataFrame to store not_selected values
            not_selected_data = {d: self.not_selected_values[d] for d in self.not_selected}

            # Merge not_selected_data with existing track data
            t2_new = pd.concat([t2, pd.DataFrame(not_selected_data)], axis=1)
            
            Q.append(t2_new)
            '''

        return Q


class Numpyfier(BaseEstimator, TransformerMixin):
    '''
    Just converts the values in a MocapData object into a numpy array
    Useful for the final stage of a pipeline before training
    '''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        #print("Numpyfier")
        Q = []
        
        for track in X:
            Q.append(np.array(track.values.values))
            #print("Numpyfier:" + str(track.values.columns))
            
        return Q   # return np.array(Q)        
        
        ##
        # for track in X:
        ##    Q.append(track.values.values)
        ##    #print("Numpyfier:" + str(track.values.columns))
        ##    
        ##return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')*self.org_mocap_.framerate

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q
    
class Slicer(BaseEstimator, TransformerMixin):
    '''
    Slice the data into intervals of equal size 
    '''
    def __init__(self, window_size, overlap=0.5):
        self.window_size = window_size
        self.overlap = overlap
        pass

    def fit(self, X, y=None):
        self.org_mocap_ = X[0].clone()
        self.org_mocap_.values.drop(self.org_mocap_.values.index, inplace=True)

        return self

    def transform(self, X, y=None):
        #print("Slicer")
        Q = []
        
        for track in X:
            vals = track.values.values
            nframes = vals.shape[0]
            overlap_frames = (int)(self.overlap*self.window_size)
            
            n_sequences = (nframes-overlap_frames)//(self.window_size-overlap_frames)
            
            if n_sequences>0:
                y = np.zeros((n_sequences, self.window_size, vals.shape[1]))

                # extract sequences from the input data
                for i in range(0,n_sequences):
                    frameIdx = (self.window_size-overlap_frames) * i
                    Q.append(vals[frameIdx:frameIdx+self.window_size,:])

        return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            
            new_mocap = self.org_mocap_.clone()
            time_index = pd.to_timedelta([f for f in range(track.shape[0])], unit='s')

            new_df =  pd.DataFrame(data=track, index=time_index, columns=self.org_mocap_.values.columns)
            
            new_mocap.values = new_df
            

            Q.append(new_mocap)

        return Q

class RootTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method, hips_axis_order="XYZ", position_smoothing=0, rotation_smoothing=0, separate_root=True):
        """
        Accepted methods:
            abdolute_translation_deltas
            pos_rot_deltas
        """
        self.method = method
        self.position_smoothing=position_smoothing
        self.rotation_smoothing=rotation_smoothing
        self.separate_root = separate_root
        self.hips_axis_order = hips_axis_order
        
        # relative rotation from the hips awis the the x-side, y-up, z-forward convention
        rot_mat = np.zeros((3,3))
        for i in range(3):
            ax_i = ord(hips_axis_order[i])-ord("X")    
            rot_mat[i,ax_i]=1
        self.root_rotation_offset = Quaternions.from_transforms(rot_mat[np.newaxis, :, :])
        self.hips_side_axis = -rot_mat[0,:]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #print("RootTransformer")
        Q = []

        for track in X:
            if self.method == 'abdolute_translation_deltas':
                new_df = track.values.copy()
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name
                
                x=track.values[xpcol].copy()
                z=track.values[zpcol].copy()
                
                if self.position_smoothing>0:
                    x_sm = filters.gaussian_filter1d(x, self.position_smoothing, axis=0, mode='nearest')    
                    z_sm = filters.gaussian_filter1d(z, self.position_smoothing, axis=0, mode='nearest')                    
                    dx = pd.Series(data=x_sm, index=new_df.index).diff()
                    dz = pd.Series(data=z_sm, index=new_df.index).diff()
                    new_df[xpcol] = x-x_sm
                    new_df[zpcol] = z-z_sm
                else:
                    dx = x.diff()
                    dz = z.diff()
                    new_df.drop([xpcol, zpcol], axis=1, inplace=True)
                    
                '''
                dx[0] = dx[1]
                dz[0] = dz[1]
                '''

                # Assume dx and dz are pandas Series objects
                dx.iloc[0] = dx.iloc[1]  # Use .iloc to set values by position
                dz.iloc[0] = dz.iloc[1]  # Similarly, apply the same operation to dz
                
                new_df[dxpcol] = dx
                new_df[dzpcol] = dz
                
                new_track = track.clone()
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name
                
                #rot_order = track.skeleton[track.root_name]['order']
                #%(joint, rot_order[0])

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name

                # dxr_col = '%s_dXrotation'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                # dzr_col = '%s_dZrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dzp_col = 'reference_dZposition'
                dxr_col = 'reference_dXrotation'
                dyr_col = 'reference_dYrotation'
                dzr_col = 'reference_dZrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                """ Get Trajectory and smooth it"""                
                trajectory_filterwidth = self.position_smoothing
                reference = positions.copy()*np.array([1,0,1])
                if trajectory_filterwidth>0:
                    reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
                
                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)                
                velocity = np.vstack((velocity[0,:], velocity))

                """ Remove Root Translation """
                positions = positions-reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                #forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                #print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)
                #forward = quats*np.array([[0,0,1]])
                #forward[:,1] = 0
                side_dirs = quats*self.hips_side_axis
                forward = np.cross(np.array([[0,1,0]]), side_dirs)

                """ Smooth Forward Direction """                
                direction_filterwidth = self.rotation_smoothing
                if direction_filterwidth>0:
                    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    

                forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

                """ Remove Y Rotation """
                target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:,np.newaxis]    
                positions = (-rotation[:,0]) * positions
                #new_rotations = (-rotation[:,0]) * quats
                new_rotations = (-self.root_rotation_offset) * (-rotation[:,0]) * quats

                """ Get Root Rotation """
                #print(rotation[:,0])
                velocity = (-rotation[:,0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in new_rotations])*180.0/np.pi
                
                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:,0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:,1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:,2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:,0], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:,2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:,0], index=new_df.index)
                
                #new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                #new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                #new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df
            elif self.method == 'pos_xyz_rot_deltas':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name
                
                #rot_order = track.skeleton[track.root_name]['order']
                #%(joint, rot_order[0])

                rot_order = track.skeleton[track.root_name]['order']
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name

                # dxr_col = '%s_dXrotation'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                # dzr_col = '%s_dZrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dyp_col = 'reference_dYposition'
                dzp_col = 'reference_dZposition'
                dxr_col = 'reference_dXrotation'
                dyr_col = 'reference_dYrotation'
                dzr_col = 'reference_dZrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                
                """ Get Trajectory and smooth it"""                
                trajectory_filterwidth = self.position_smoothing
                #reference = positions.copy()*np.array([1,0,1])
                if trajectory_filterwidth>0:
                    reference = filters.gaussian_filter1d(positions, trajectory_filterwidth, axis=0, mode='nearest')
                
                """ Get Root Velocity """
                velocity = np.diff(reference, axis=0)                
                velocity = np.vstack((velocity[0,:], velocity))

                """ Remove Root Translation """
                positions = positions-reference

                """ Get Forward Direction along the x-z plane, assuming character is facig z-forward """
                #forward = [Rotation(f, 'euler', from_deg=True, order=rot_order).rotmat[:,2] for f in rotations] # get the z-axis of the rotation matrix, assuming character is facig z-forward
                #print("order:" + rot_order.lower())
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                #calculate the hips forward directions given in global cordinates 
                #side_ax = np.zeros((1,3))
                #side_ax[0,self.hips_side_axis]=1
                #side_dirs = quats*side_ax
                side_dirs = quats*self.hips_side_axis
                forward = np.cross(np.array([[0,1,0]]), side_dirs)

                """ Smooth Forward Direction """                
                direction_filterwidth = self.rotation_smoothing
                if direction_filterwidth>0:
                    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')    

                # make unit vector
                forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

                """ Remove Y Rotation """
                target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
                rotation = Quaternions.between(target, forward)[:,np.newaxis]    
                positions = (-rotation[:,0]) * positions
                new_rotations = (-self.root_rotation_offset) * (-rotation[:,0]) * quats

                """ Get Root Rotation """
                #print(rotation[:,0])
                velocity = (-rotation[:,0]) * velocity
                rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps
                rvelocity = np.vstack((rvelocity[0], rvelocity))

                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in new_rotations])*180.0/np.pi
                
                new_df = track.values.copy()

                root_pos_x = pd.Series(data=positions[:,0], index=new_df.index)
                root_pos_y = pd.Series(data=positions[:,1], index=new_df.index)
                root_pos_z = pd.Series(data=positions[:,2], index=new_df.index)
                root_pos_x_diff = pd.Series(data=velocity[:,0], index=new_df.index)
                root_pos_y_diff = pd.Series(data=velocity[:,1], index=new_df.index)
                root_pos_z_diff = pd.Series(data=velocity[:,2], index=new_df.index)

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                root_rot_y_diff = pd.Series(data=rvelocity[:,0], index=new_df.index)
                
                #new_df.drop([xr_col, yr_col, zr_col, xp_col, zp_col], axis=1, inplace=True)

                new_df[xp_col] = root_pos_x
                new_df[yp_col] = root_pos_y
                new_df[zp_col] = root_pos_z
                new_df[dxp_col] = root_pos_x_diff
                new_df[dyp_col] = root_pos_y_diff
                new_df[dzp_col] = root_pos_z_diff

                new_df[r1_col] = root_rot_1
                new_df[r2_col] = root_rot_2
                new_df[r3_col] = root_rot_3
                #new_df[dxr_col] = root_rot_x_diff
                new_df[dyr_col] = root_rot_y_diff
                #new_df[dzr_col] = root_rot_z_diff

                new_track.values = new_df


            elif self.method == 'hip_centric':
                new_track = track.clone()

                # Absolute columns
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                
                new_df = track.values.copy()

                all_zeros = np.zeros(track.values[xp_col].values.shape)

                new_df[xp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zp_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_df[xr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[yr_col] = pd.Series(data=all_zeros, index=new_df.index)
                new_df[zr_col] = pd.Series(data=all_zeros, index=new_df.index)

                new_track.values = new_df

            #print(new_track.values.columns)
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None, start_pos=None):
        Q = []

        #TODO: simplify this implementation

        startx = 0
        startz = 0

        if start_pos is not None:
            startx, startz = start_pos

        for track in X:
            new_track = track.clone()
            if self.method == 'abdolute_translation_deltas':
                new_df = new_track.values
                xpcol = '%s_Xposition'%track.root_name
                ypcol = '%s_Yposition'%track.root_name
                zpcol = '%s_Zposition'%track.root_name


                dxpcol = '%s_dXposition'%track.root_name
                dzpcol = '%s_dZposition'%track.root_name

                dx = track.values[dxpcol].values
                dz = track.values[dzpcol].values

                recx = [startx]
                recz = [startz]

                for i in range(dx.shape[0]-1):
                    recx.append(recx[i]+dx[i+1])
                    recz.append(recz[i]+dz[i+1])

                # recx = [recx[i]+dx[i+1] for i in range(dx.shape[0]-1)]
                # recz = [recz[i]+dz[i+1] for i in range(dz.shape[0]-1)]
                # recx = dx[:-1] + dx[1:]
                # recz = dz[:-1] + dz[1:]
                if self.position_smoothing > 0:                    
                    new_df[xpcol] = pd.Series(data=new_df[xpcol]+recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=new_df[zpcol]+recz, index=new_df.index)
                else:
                    new_df[xpcol] = pd.Series(data=recx, index=new_df.index)
                    new_df[zpcol] = pd.Series(data=recz, index=new_df.index)

                new_df.drop([dxpcol, dzpcol], axis=1, inplace=True)
                
                new_track.values = new_df
            # end of abdolute_translation_deltas
            
            elif self.method == 'pos_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dzp_col = 'reference_dZposition'
                dyr_col = 'reference_dYrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                #rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_zp = [0]

                #rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0]-1):
                    #print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i+1]), np.array([0,1,0]))
                    rec_ry[i+1] = q_y*rec_ry[i]
                    #print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i+1]*np.array([dx[i+1], 0, dz[i+1]])
                    rec_xp.append(rec_xp[i]+dp[0,0])
                    rec_zp.append(rec_zp[i]+dp[0,2])
                    
                if self.separate_root:
                    qq = quats
                    xx = positions[:,0]
                    zz = positions[:,2]
                else:
                    qq = rec_ry*self.root_rotation_offset*quats
                    pp = rec_ry*positions
                    xx = rec_xp + pp[:,0]
                    zz = rec_zp + pp[:,2]
                
                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in qq])*180.0/np.pi
                
                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                
                new_df[xp_col] = pd.Series(data=xx, index=new_df.index)
                new_df[zp_col] = pd.Series(data=zz, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                if self.separate_root:
                    ref_rot_order="ZXY"
                    new_df["reference_Xposition"] = pd.Series(data=rec_xp, index=new_df.index)
                    new_df["reference_Zposition"] = pd.Series(data=rec_zp, index=new_df.index)                    
                    eulers_ry = np.array([t3d.euler.quat2euler(q, axes=('s'+ref_rot_order.lower()[::-1]))[::-1] for q in rec_ry])*180.0/np.pi
                    new_df["reference_Yrotation"] = pd.Series(data=eulers_ry[:,ref_rot_order.find('Y')], index=new_df.index)
                    

                new_df.drop([dyr_col, dxp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df
                
            elif self.method == 'pos_xyz_rot_deltas':
                # Absolute columns
                rot_order = track.skeleton[track.root_name]['order']
                xp_col = '%s_Xposition'%track.root_name
                yp_col = '%s_Yposition'%track.root_name
                zp_col = '%s_Zposition'%track.root_name

                xr_col = '%s_Xrotation'%track.root_name
                yr_col = '%s_Yrotation'%track.root_name
                zr_col = '%s_Zrotation'%track.root_name
                r1_col = '%s_%srotation'%(track.root_name, rot_order[0])
                r2_col = '%s_%srotation'%(track.root_name, rot_order[1])
                r3_col = '%s_%srotation'%(track.root_name, rot_order[2])

                # Delta columns
                # dxp_col = '%s_dXposition'%track.root_name
                # dzp_col = '%s_dZposition'%track.root_name
                # dyr_col = '%s_dYrotation'%track.root_name
                dxp_col = 'reference_dXposition'
                dyp_col = 'reference_dYposition'
                dzp_col = 'reference_dZposition'
                dyr_col = 'reference_dYrotation'

                positions = np.transpose(np.array([track.values[xp_col], track.values[yp_col], track.values[zp_col]]))
                rotations = np.pi/180.0*np.transpose(np.array([track.values[r1_col], track.values[r2_col], track.values[r3_col]]))
                quats = Quaternions.from_euler(rotations, order=rot_order.lower(), world=False)

                new_df = track.values.copy()

                dx = track.values[dxp_col].values
                dy = track.values[dyp_col].values
                dz = track.values[dzp_col].values

                dry = track.values[dyr_col].values

                #rec_p = np.array([startx, 0, startz])+positions[0,:]
                rec_ry = Quaternions.id(quats.shape[0])
                rec_xp = [0]
                rec_yp = [0]
                rec_zp = [0]

                #rec_r = Quaternions.id(quats.shape[0])

                for i in range(dx.shape[0]-1):
                    #print(dry[i])
                    q_y = Quaternions.from_angle_axis(np.array(dry[i+1]), np.array([0,1,0]))
                    rec_ry[i+1] = q_y*rec_ry[i]
                    #print("dx: + " + str(dx[i+1]))
                    dp = rec_ry[i+1]*np.array([dx[i+1], dy[i+1], dz[i+1]])
                    rec_xp.append(rec_xp[i]+dp[0,0])
                    rec_yp.append(rec_yp[i]+dp[0,1])
                    rec_zp.append(rec_zp[i]+dp[0,2])
                    
                if self.separate_root:
                    qq = quats
                    xx = positions[:,0]
                    yy = positions[:,1]
                    zz = positions[:,2]
                else:
                    qq = rec_ry*self.root_rotation_offset*quats
                    pp = rec_ry*positions
                    xx = rec_xp + pp[:,0]
                    yy = rec_yp + pp[:,1]
                    zz = rec_zp + pp[:,2]
                
                eulers = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in qq])*180.0/np.pi
                
                new_df = track.values.copy()

                root_rot_1 = pd.Series(data=eulers[:,0], index=new_df.index)
                root_rot_2 = pd.Series(data=eulers[:,1], index=new_df.index)
                root_rot_3 = pd.Series(data=eulers[:,2], index=new_df.index)
                
                new_df[xp_col] = pd.Series(data=xx, index=new_df.index)
                new_df[yp_col] = pd.Series(data=yy, index=new_df.index)
                new_df[zp_col] = pd.Series(data=zz, index=new_df.index)

                new_df[r1_col] = pd.Series(data=root_rot_1, index=new_df.index)
                new_df[r2_col] = pd.Series(data=root_rot_2, index=new_df.index)
                new_df[r3_col] = pd.Series(data=root_rot_3, index=new_df.index)

                if self.separate_root:
                    new_df["reference_Xposition"] = pd.Series(data=rec_xp, index=new_df.index)
                    new_df["reference_Yposition"] = pd.Series(data=rec_yp, index=new_df.index)
                    new_df["reference_Zposition"] = pd.Series(data=rec_zp, index=new_df.index)                    
                    eulers_ry = np.array([t3d.euler.quat2euler(q, axes=('s'+rot_order.lower()[::-1]))[::-1] for q in rec_ry])*180.0/np.pi
                    new_df["reference_Yrotation"] = pd.Series(data=eulers_ry[:,rot_order.find('Y')], index=new_df.index)
                    

                new_df.drop([dyr_col, dxp_col, dyp_col, dzp_col], axis=1, inplace=True)


                new_track.values = new_df
                
            #print(new_track.values.columns)
            Q.append(new_track)

        return Q



class RootCentricPositionNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            all_but_root = [joint for joint in track.skeleton if track.root_name not in joint]
            # all_but_root = [joint for joint in track.skeleton]
            '''
            for joint in all_but_root:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]-projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]-projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]-projected_root_pos[rzp], index=new_df.index)
            '''
            joint_positions = {
                 col[-10:]: track.values[col] - projected_root_pos[col[-10:]] for col in track.values.keys()
                 if col.endswith('_position')
             }
            new_df = pd.concat([new_df, pd.DataFrame(joint_positions)], axis=1)
                
            # keep the root as it is now
            new_df[rxp] = track.values[rxp]
            new_df[ryp] = track.values[ryp]
            new_df[rzp] = track.values[rzp]

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

    def inverse_transform(self, X, copy=None):
        Q = []

        for track in X:
            new_track = track.clone()

            rxp = '%s_Xposition'%track.root_name
            ryp = '%s_Yposition'%track.root_name
            rzp = '%s_Zposition'%track.root_name

            projected_root_pos = track.values[[rxp, ryp, rzp]]

            projected_root_pos.loc[:,ryp] = 0 # we want the root's projection on the floor plane as the ref

            new_df = pd.DataFrame(index=track.values.index)

            for joint in track.skeleton:                
                new_df['%s_Xposition'%joint] = pd.Series(data=track.values['%s_Xposition'%joint]+projected_root_pos[rxp], index=new_df.index)
                new_df['%s_Yposition'%joint] = pd.Series(data=track.values['%s_Yposition'%joint]+projected_root_pos[ryp], index=new_df.index)
                new_df['%s_Zposition'%joint] = pd.Series(data=track.values['%s_Zposition'%joint]+projected_root_pos[rzp], index=new_df.index)
                

            new_track.values = new_df

            Q.append(new_track)
        
        return Q

class Flattener(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.concatenate(X, axis=0)

class ConstantsRemover(BaseEstimator, TransformerMixin):
    '''
    This class is used to remove constant features from datasets.
    It implements scikit-learn's BaseEstimator and TransformerMixin interfaces for use in scikit-learn pipelines.
    '''
    '''
    For now it just looks at the first track
    '''

    def __init__(self, eps=1e-6):
        '''
        Initialize the ConstantsRemover object.

        Args:
        eps: float
            Threshold for determining if a feature is constant. If a feature's standard deviation is less than this value,
            it is considered a constant feature.
        '''
        self.eps = eps
        
    def fit(self, X, y=None):
        '''
        Identify constant features in the dataset.

        Args:
        X: list of pandas.DataFrame
            The dataset, where each element is a pandas.DataFrame.
        y: None
            Labels, currently unused but kept for scikit-learn interface compatibility.

        Returns:
        self: ConstantsRemover
            Returns the fitted object itself.
        '''
        stds = X[0].values.std()
        cols = X[0].values.columns.values
        # Mark features with standard deviation less than eps as constant features
        self.const_dims_ = [c for c in cols if (stds[c] < self.eps)]
        # Store the constant feature values
        self.const_values_ = {c: X[0].values[c].values[0] for c in cols if (stds[c] < self.eps)}
        return self
    
    def transform(self, X, y=None):
        '''
        Remove constant features from the dataset.

        Args:
        X: list of pandas.DataFrame
            The dataset, where each element is a pandas.DataFrame.
        y: None
            Labels, currently unused but kept for scikit-learn interface compatibility.

        Returns:
        Q: list
            New dataset list with constant features removed.
        '''
        Q = []
        for track in X:
            t2 = track.clone()
            # Remove constant features
            t2.values.drop(self.const_dims_, axis=1, inplace=True)
            Q.append(t2)
        return Q

    def inverse_transform(self, X, copy=None):
        '''
        Add previously removed constant features back to the dataset.

        Args:
        X: list of pandas.DataFrame
            The dataset, where each element is a pandas.DataFrame.
        copy: bool or None
            Whether to copy the dataset, defaults to None (no copy).

        Returns:
        Q: list
            New dataset list with constant features added back.
        '''
        Q = []
        for track in X:
            t2 = track.clone()
            # Add constant features back to the dataset
            for d in self.const_dims_:
                t2.values[d] = self.const_values_[d]
            Q.append(t2)
        return Q

class ListStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_mean_ = np.mean(X_train_flat, axis=0)
        self.data_std_ = np.std(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_mean_) / self.data_std_
            else:
                normalized_track = (track - self.data_mean_) / self.data_std_

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * self.data_std_) + self.data_mean_
            else:
                unnormalized_track = (track * self.data_std_) + self.data_mean_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

class ListMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, is_DataFrame=False):
        self.is_DataFrame = is_DataFrame
    
    def fit(self, X, y=None):
        if self.is_DataFrame:
            X_train_flat = np.concatenate([m.values for m in X], axis=0)
        else:
            X_train_flat = np.concatenate([m for m in X], axis=0)

        self.data_max_ = np.max(X_train_flat, axis=0)
        self.data_min_ = np.min(X_train_flat, axis=0)

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            if self.is_DataFrame:
                normalized_track = track.copy()
                normalized_track.values = (track.values - self.data_min_) / (self.data_max_ - self.data_min_) 
            else:
                normalized_track = (track - self.data_min_) / (self.data_max_ - self.data_min_)

            Q.append(normalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)

    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            
            if self.is_DataFrame:
                unnormalized_track = track.copy()
                unnormalized_track.values = (track.values * (self.data_max_ - self.data_min_)) + self.data_min_
            else:
                unnormalized_track = (track * (self.data_max_ - self.data_min_)) + self.data_min_

            Q.append(unnormalized_track)
        
        if self.is_DataFrame:
            return Q
        else:
            return np.array(Q)
        
class Resampler(BaseEstimator, TransformerMixin):
    def __init__(self, fps, method='cubic'):
        '''
        Method to resample a pandas dataframe to a different framerate.
        NOTE: Pandas resampling is quit unintuitive when resampling to odd framerates using interpolation.
        Thus we do it in this complex way.
        '''
        self.tgt_frametime = 1.0/fps
        self.method = method
    
    def fit(self, X, y=None):
        #print("Resampling to tgt_frametime: " + str(self.tgt_frametime))
        self.orig_frametime=X[0].framerate
        return self
    
    def resample_dataframe(self, df, frametime, method='cubic'):
        #Create a time index for the resampled data
        rate = str(round(1.0e9*frametime))+'N'
        time_index = df.resample(rate).indices
        
        #reindex the old data. This will turn all non-matching indices to NAN
        tmp = df.reindex(time_index)
        
        #merge with the old data and sort
        tmp = pd.concat([df, tmp]).sort_index()
        
        #remove duplicate time indices. Then fill the NAN values using interpolation
        tmp=tmp[~tmp.index.duplicated(keep='first')].interpolate(method=method)

        #return the values using the resampled indices
        return tmp.loc[list(time_index)]
        
    def resample_df(self, df, new_frametime, old_frametime, mode='cubic'):

        #Create a time index for the resampled data
        data = df.values

        nframes = data.shape[0]
        nframes_new = round(nframes*old_frametime/new_frametime)
        x = np.arange(0, nframes)/(nframes-1)
        xnew = np.arange(0, nframes_new)/(nframes_new-1)

        data_out = np.zeros((nframes_new, data.shape[1]))
        for jj in range(data.shape[1]):
            y = data[:,jj]
            f = interpolate.interp1d(x, y, bounds_error=False, kind=mode, fill_value='extrapolate')
            data_out[:,jj] = f(xnew)

        time_index = pd.to_timedelta([f for f in range(xnew.shape[0])], unit='s')*new_frametime
        out = pd.DataFrame(data=data_out, index=time_index, columns=df.columns)
        
        #Scale root deltas to match new frame-rate
        sc = nframes/nframes_new
        rootdelta_cols = [c for c in df.columns if ('reference_d' in c)]    
        out[rootdelta_cols]*=sc

        return out 

    # def resample_poly_df(self, df, new_frametime, old_frametime):
        # old_fps = round(1/old_frametime)
        # new_fps = round(1/new_frametime)
        # lcm = np.lcm(old_fps, new_fps)
        # up = lcm//old_fps
        # down = lcm//new_fps
        # new_vals = signal.resample_poly(df.values, up, down, padtype='line')
        # time_index = pd.to_timedelta([f for f in range(new_vals.shape[0])], unit='s')*new_frametime
        # new_df =  pd.DataFrame(data=new_vals, index=time_index, columns=df.columns)
        # return new_df
        
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            new_track = track.clone()
            # if self.method=="resample_poly":
                # new_track.values = self.resample_poly_df(track.values, self.tgt_frametime, track.framerate)
            # else:
            new_track.values = self.resample_df(track.values, self.tgt_frametime, track.framerate, self.method)
            #new_track.values = self.resample_dataframe(track.values, self.tgt_frametime, method=self.method)
            new_track.framerate = self.tgt_frametime
            Q.append(new_track)
        
        return Q
        
    def inverse_transform(self, X, copy=None):
        Q = []
        
        for track in X:
            new_track = track.clone()
            #new_track.values = self.resample_dataframe(track.values, self.orig_frametime, method=self.method)
            if self.method=="resample_poly":
                new_track.values = self.resample_poly_df(track.values, self.orig_frametime, track.framerate)
            else:
                new_track.values = self.resample_df(track.values, self.orig_frametime, track.framerate, self.method)
            new_track.framerate = self.orig_frametime
            Q.append(new_track)
        
        return Q


class DownSampler(BaseEstimator, TransformerMixin):
    def __init__(self, tgt_fps, keep_all=False):
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            orig_fps=round(1.0/track.framerate)
            rate = orig_fps//self.tgt_fps
            if orig_fps%self.tgt_fps!=0:
                print("error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            else:
                print("downsampling with rate: " + str(rate))
                
            #print(track.values.size)
            for ii in range(0,rate):
                new_track = track.clone()
                if self.keep_all:
                    new_track.take_name = new_track.take_name + "_" + str(ii).zfill(2)
                new_track.values = track.values[ii::rate].copy()            
                #print(new_track.values.size)
                #new_track = track[0:-1:self.rate]
                new_track.framerate = 1.0/self.tgt_fps
                Q.append(new_track)
                if not self.keep_all:
                    break
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X

'''
class DownSampler(BaseEstimator, TransformerMixin):
    """

    DownSampler has major issues - track.clone and framerate cannot be found during transformation
    """
    
    def __init__(self, framerate, tgt_fps, keep_all=False):
        self.framerate = framerate
        self.tgt_fps = tgt_fps
        self.keep_all = keep_all
        
    
    def fit(self, X, y=None):    

        return self
    
    def transform(self, X, y=None):
        Q = []
        
        for track in X:
            print(track)
            
            
            orig_fps=round(1.0/self.framerate) # track.framerate
            print("orig_fps: " + str(orig_fps))
            rate = orig_fps//self.tgt_fps
            
            if orig_fps%self.tgt_fps!=0:
                print("error orig_fps (" + str(orig_fps) + ") is not dividable with tgt_fps (" + str(self.tgt_fps) + ")")
            else:
                print("downsampling with rate: " + str(rate))
                
            #print(track.values.size)
            for ii in range(0,rate):
                new_track = track.copy()
                if self.keep_all:
                    new_track.take_name = new_track.take_name + "_" + str(ii).zfill(2)
                new_track = track[ii::rate].copy()
                                                        
                ## Error fix: new_track.values = track.values[ii::rate].copy()            
                #print(new_track.values.size)
                #new_track = track[0:-1:self.rate]
                
                ###new_track.framerate = 1.0/self.tgt_fps
                
                Q.append(new_track)
                if not self.keep_all:
                    break
        
        return Q
        
    def inverse_transform(self, X, copy=None):
      return X
'''

class ReverseTime(BaseEstimator, TransformerMixin):
    """
    The ReverseTime class is used to reverse time series data.

    Inherits from BaseEstimator and TransformerMixin to implement transformation operations in machine learning pipelines.

    Args:
    - append: bool, default True. Controls whether to append the reversed time series after the original time series.
    """

    def __init__(self, append=True):
        """
        Initialize the ReverseTime class instance.
        """
        self.append = append
        
    def fit(self, X, y=None):
        """
        Learning phase, this method is empty and only exists to comply with the TransformerMixin interface.

        Args:
        - X: Input data, type is list, each element is a time series.
        - y: Label data, not needed in this method, can be None.

        Returns:
        - self: The class instance itself, for chain calling.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Perform reversal processing on time series.

        Args:
        - X: Input data, type is list, each element is a time series.
        - y: Label data, not needed in this method, can be None.

        Returns:
        - Q: List of reversed time series. If append is True, includes both original and reversed series.
        """
        # Initialize result list
        Q = []
        # If append is True, add original series to result list
        if self.append:
            for track in X:
                Q.append(track)
        # Perform reversal processing on each time series
        for track in X:
            # Clone original series to keep original data unchanged
            new_track = track.clone()
            # Reverse time series values
            new_track.values = track.values[-1::-1]
            # Adjust the index of reversed time series
            new_track.values.index = new_track.values.index[0] - new_track.values.index
            # Add reversed series to result list
            Q.append(new_track)

        return Q

    def inverse_transform(self, X, copy=None):
        """
        Perform inverse operation on reversed time series, i.e., restore original series.

        Since reversal is reversible, this method directly returns the input X.

        Args:
        - X: List of reversed time series.
        - copy: bool, controls whether to copy input data, defaults to None.

        Returns:
        - X: Input time series list, unmodified.
        """
        
        return X

class ListFeatureUnion(BaseEstimator, TransformerMixin):
    """
    The ListFeatureUnion class implements feature union transformation.

    This class achieves parallel processing and feature merging by combining multiple feature processing pipelines.
    It is a custom implementation of sklearn's FeatureUnion concept, suitable for processing DataFrames in lists.

    Args:
    processors: list
        List of feature processing pipelines to combine.
    """

    def __init__(self, processors):
        """
        Initialize ListFeatureUnion instance.

        Args:
        processors: list
            List of feature processing pipelines to combine.
        """
        self.processors = processors
    
    def fit(self, X, y=None):
        """
        Fit all processors.

        For each processor, if it is a Pipeline instance, recursively fit each step;
        otherwise, fit the processor directly. Ignore y value.

        Args:
        X: Input data.
        y: Label data (currently unused but kept for compatibility).

        Returns:
        self: Fitted instance.
        """
        assert(y is None)
        for proc in self.processors:
            if isinstance(proc, Pipeline):
                #Loop steps and run fit on each. This is necessary since
                #running fit on a Pipeline runs fit_transform on all steps
                #and not only fit.
                for step in proc.steps:
                    step[1].fit(X)
            else:
                proc.fit(X)
        return self
    
    def transform(self, X, y=None):
        """
        Perform feature transformation on input data.

        Merge outputs from each processor. If this is the first processor, assign directly;
        otherwise, ensure the current processor's output has the same length as previous processors' outputs, then merge.

        Args:
        X: Input data.
        y: Label data (currently unused but kept for compatibility).

        Returns:
        Q: Data transformed by all processors.
        """
        assert(y is None)
        #print("ListFeatureUnion")
        Q = []
        idx = 0
        for proc in self.processors:
            Z = proc.transform(X)
            if idx == 0:
                Q = Z
            else:
                assert(len(Q) == len(Z))
                for idx2, track in enumerate(Z):
                    Q[idx2].values = pd.concat([Q[idx2].values, Z[idx2].values], axis=1)
            idx += 1
        return Q
    
    def inverse_transform(self, X, y=None):
        """
        Convert transformed data back to original form.

        Since this implementation doesn't perform actual transformation (transform method only merges features),
        the inverse transform operation directly returns the input data.

        Args:
        X: Transformed data.
        y: Label data (currently unused but kept for compatibility).

        Returns:
        X: Original input data.
        """
        return X

class RollingStatsCalculator(BaseEstimator, TransformerMixin):
    """
    Class for computing rolling statistical features. This class inherits from sklearn's BaseEstimator and TransformerMixin,
    used to compute rolling mean and standard deviation for time series with a given window size.

    Args:
    win: int
        Size of the rolling window.
    """
    '''
    Creates a causal mean and std filter with a rolling window of length win (based on using prev and current values)
    '''
    def __init__(self, win):
        self.win = win
            
    def fit(self, X, y=None):
        """
       Fit the transformer. This method is part of the sklearn interface, but doesn't need to perform any operations in this class.

        Args:
        X: pandas.DataFrame
            Input data.
        y: None
            This parameter is not needed in this method, so set to None.

        Returns:
        self: RollingStatsCalculator
            Returns self to support chain calling.
        """
        return self
    
    def transform(self, X, y=None):
        """
        Compute rolling mean and standard deviation.

        Args:
        X: pandas.DataFrame
            Input time series data.
        y: None
            This parameter is not needed in this method, so set to None.

        Returns:
        pandas.DataFrame
            New time series containing rolling mean and standard deviation.
        """
        #print("RollingStatsCalculator: " + str(self.win))

        Q = []
        for track in X:
            new_track = track.clone()
            # Compute rolling mean
            mean_df = track.values.rolling(window=self.win).mean()
            # Compute rolling standard deviation
            std_df = track.values.rolling(window=self.win).std()
            # Handle cases where window size is less than win, by forward-filling computed mean and std
            win = min(self.win, new_track.values.shape[0])
            for i in range(1, win):
                mm = track.values[:i].rolling(window=i).mean()
                ss = track.values[:i].rolling(window=i).std()
                mean_df.iloc[i-1] = mm.iloc[i-1]
                std_df.iloc[i-1] = ss.iloc[i-1]

            # Handle the first value's standard deviation by copying the second value
            std_df.iloc[0] = std_df.iloc[1]
            # Merge mean and standard deviation into a new time series
            new_track.values = pd.concat([mean_df.add_suffix('_mean'), std_df.add_suffix('_std')], axis=1)
            Q.append(new_track)
        return Q

    def inverse_transform(self, X, copy=None):
        """
        Perform inverse transform on given data. Since this class only computes new features and doesn't modify original data,
        the inverse transform simply returns the input data.

        Args:
        X: pandas.DataFrame
            Input data.
        copy: None
            This parameter is not needed in this method, so set to None.

        Returns:
        pandas.DataFrame
            Returns the input data X.
        """
        return X

class FeatureCounter(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.n_features = len(X[0].values.columns)
        print("FeatureCounter: " + str(self.n_features))
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, copy=None):
        return X

#TODO: JointsSelector (x)
#TODO: SegmentMaker
#TODO: DynamicFeaturesAdder
#TODO: ShapeFeaturesAdder
#TODO: DataFrameNumpier (x)

class TemplateTransform(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

