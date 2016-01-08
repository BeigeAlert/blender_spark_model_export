# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

# Animation-related code

import bpy
import sys
from spark_common import *
from export_spark_model import *


# Constants
def ANIMATION_FLAG_RELATIVE(): return 1
def ANIMATION_FLAG_LOOPING(): return 2

def ANIMATION_NODE_TYPE_ANIMATION(): return 1
def ANIMATION_NODE_TYPE_BLEND(): return 2
def ANIMATION_NODE_TYPE_LAYER(): return 3


class FrameTag:
    def __init__(self, *args):
        self.frame = -1
        """:type : int"""

        self.name = ''
        """:type : str"""

        if len(args) == 1:  # copy of existing frame tag
            old_tag = args[0]
            self.frame = old_tag.frame
            self.name = old_tag.name


def eval_error(expected, actual):
    if isinstance(expected, Vec3):
        if not isinstance(actual, Vec3):
            raise SparkException("EvalError requires expected and actual be like-classes.")
        return (expected - actual).get_length()
    elif isinstance(expected, Quat):
        if not isinstance(actual, Quat):
            raise SparkException("EvalError requires expected and actual be like-classes.")
        return expected.get_distance(actual)
    else:
        try:
            x = float(expected)
            y = float(actual)
            return abs(x - y)
        except:
            raise SparkException("EvalError expects a pair of Vec3, a pair of Quat, or a pair of numerical values.")


def interpolate_poses(p1, p2, f):
    if isinstance(p1, Vec3):
        if not isinstance(p2, Vec3):
            raise TypeError("Cannot interpolate Vec3 with non-Vec3 type.")
        return ((p2 - p1) * f) + p1
    elif isinstance(p1, Quat):
        if not isinstance(p2, Quat):
            raise TypeError("Cannot interpolate Quat with non-Quat type.")

        if p1.dot_product(p2) < 0.0:
            s = -(1.0 - f)
        else:
            s = 1.0 - f
        
        result = Quat()
        
        result.x = p1.x * s + p2.x * f
        result.y = p1.y * s + p2.y * f
        result.z = p1.z * s + p2.z * f
        result.w = p1.w * s + p2.w * f
        
        k = 1.0 / math.sqrt(result.dot_product(result))
        
        result.x *= k
        result.y *= k
        result.z *= k
        result.w *= k
        
        return result
    else:
        try:
            v1 = float(p1)
            v2 = float(p2)
            return ((v2 - v1) * f) + v1
        except:
            raise TypeError("Expected numerical values for both inputs")


class CurveFitter:
    def __init__(self, bone_animation, compression_settings, curve_type=0):
        self.type = curve_type
        """:type : int"""
        self.compression_settings = compression_settings
        """:type : CompressionSettings"""
        self.bone_animation = bone_animation
        """:type : BoneAnimation"""

        # x-axis is time, in seconds, not frames!
        self.min_x = 0.0
        """:type : float"""
        self.max_x = len(bone_animation) / compression_settings.frame_rate
        """:type : float"""
        self.c_keys_x = []  # compressed curve keys x (time) axis
        """:type : list[float]"""
        self.c_keys_y = []  # compressed curve keys y (value) axis
        """:type : list[Quat | Vec3 | float]"""
        
        if abs(self.max_x - self.min_x) < 0.000001:  # no frames, no need to compress
            self.c_keys_x.append(self.min_x)
            self.c_keys_y.append(self.eval(self.min_x))
        else:

            self.c_keys_x.append(self.min_x)
            self.c_keys_y.append(self.eval(self.min_x))

            self.c_keys_x.append(self.max_x)
            self.c_keys_y.append(self.eval(self.max_x))

            max_error = compression_settings.linear_max_error if self.type & 1 else compression_settings.quat_max_error
            
            seg_error = [self.eval_max_error(0)]
            worst_seg = 0
            
            while seg_error[worst_seg] > max_error:
                pid1 = worst_seg
                pid2 = worst_seg + 1
                
                if pid1 >= len(self.c_keys_x):
                    raise SparkException("worst_seg outside bounds")
                if pid2 >= len(self.c_keys_x):
                    raise SparkException("worst_seg+1 outside bounds")
                
                x1 = self.c_keys_x[pid1]
                x2 = self.c_keys_x[pid2]
                
                x = (x1 + x2) / 2.0
                self.c_keys_x.insert(pid1 + 1, x)
                self.c_keys_y.insert(pid1 + 1, self.eval(x))
                
                left_seg = worst_seg
                right_seg = left_seg + 1
                
                seg_error.insert(right_seg, 0.0)  # dummy entry
                
                # re-evaluate error for affected segments
                
                support = self.type & 1  # 1 extra frame front and back for hermite-interpolants,
                # 0 extra frames front and back for quat

                seg_first = max(left_seg - support, 0)
                seg_last = min(right_seg + support, len(seg_error) - 1)
                
                for i in range(seg_first, seg_last + 1):
                    seg_error[i] = self.eval_max_error(i)
                
                worst_seg_error = seg_error[0]
                worst_seg = 0
                for i in range(1, len(seg_error)):
                    if seg_error[i] > worst_seg_error:
                        worst_seg = i
    
    def eval(self, x):  # evaluates the animation at value x
        if x > self.max_x:
            raise SparkException("attempted to evaluate curve outside time bounds (x > max_x)")
        if x < self.min_x:
            raise SparkException("attempted to evaluate curve outside time bounds (x < min_x)")
            
        time = max(min(x, self.max_x), self.min_x)
        frame = time * self.compression_settings.frame_rate
        fraction = frame - math.floor(frame)
        frame1 = int(min(max(frame, 0), len(self.bone_animation) - 1))
        frame2 = int(min(max(frame1 + 1, 0), len(self.bone_animation) - 1))

        if self.type == 1:  # position
            return interpolate_poses(self.bone_animation[frame1].translation,
                                     self.bone_animation[frame2].translation, fraction)
        elif self.type == 2:  # rotation
            return interpolate_poses(self.bone_animation[frame1].rotation,
                                     self.bone_animation[frame2].rotation, fraction)
        elif self.type == 3:  # scale
            return interpolate_poses(self.bone_animation[frame1].scale,
                                     self.bone_animation[frame2].scale, fraction)
        elif self.type == 4:  # scale-rotation
            return interpolate_poses(self.bone_animation[frame1].scale_rotation,
                                     self.bone_animation[frame2].scale_rotation, fraction)
        elif self.type == 5:  # flip
            return interpolate_poses(self.bone_animation[frame1].flip,
                                     self.bone_animation[frame2].flip, fraction)
        else:
            raise SparkException("Invalid type of pose curve. (" + str(self.type) + ")")

    def implicit_hermite_interpolate(self, seg, t):
        n = len(self.c_keys_x)
        if t < 0.0 or t > 1.0:
            raise SparkException("Interpolation fraction must be between 0.0 and 1.0 inclusive")
        if seg < 0 or seg >= n:
            raise SparkException("Segment out of bounds")
        ym1 = self.c_keys_y[max(0, seg - 1)]
        y = self.c_keys_y[seg]
        yp1 = self.c_keys_y[min(n - 1, seg + 1)]
        yp2 = self.c_keys_y[min(n - 1, seg + 2)]
        
        # For the end cases, extend the data to keep the linear slope constant
        
        if seg == 0:
            ym1 = y - (yp1 - y)
        elif seg == n - 2:
            yp2 = yp1 + (yp1 - y)
        elif seg == n - 1:
            yp1 = y + (y - ym1)
            yp2 = yp1 + (y - ym1)
        
        ts = t * t
        ht = t / 2.0
        htc = ts * ht
        return (-htc + ts - ht) * ym1 + (3.0 * htc - 5.0 * ts / 2.0 + 1.0) * y +\
               (-3.0 * htc + 2.0 * ts + ht) * yp1 + (htc - ts / 2.0) * yp2

    def linear_quat_interpolate(self, seg, t):
        a = self.c_keys_y[seg]
        b = self.c_keys_y[seg + 1]

        if a.dot_product(b) < 0.0:
            s = t - 1.0
        else:
            s = 1.0 - t
        
        result = Quat()
        
        result.x = a.x * s + b.x * t
        result.y = a.y * s + b.y * t
        result.z = a.z * s + b.z * t
        result.w = a.w * s + b.w * t
        
        k = 1.0 / math.sqrt(result.dot_product(result))
        
        result.x *= k
        result.y *= k
        result.z *= k
        result.w *= k
        
        return result

    def interpolate(self, seg, t):
        # ImplicitHermiteInterpolate for Vec3 and Real, LinearQuatInterpolate for Quats
        if self.type & 1:  # type is 1, 3, or 5 (or 7..., or 9...)
            return self.implicit_hermite_interpolate(seg, t)
        else:  # type is 2 or 4, a quaternion
            return self.linear_quat_interpolate(seg, t)

    def eval_max_error(self, seg):
        step = self.compression_settings.sampling_period
        x1 = self.c_keys_x[seg]
        x2 = self.c_keys_x[seg + 1]
        
        max_error = -1.0
        
        if abs(x1 - x2) < 0.0000001:
            raise SparkException("Segment keys too tight!  Something went wrong.")
        
        x = x1
        while x <= x2:
            expected = self.eval(x)
            t = (x - x1) / (x2 - x1)
            actual = self.interpolate(seg, t)
            err = eval_error(expected, actual)
            max_error = max(max_error, err)
            
            x += step
        
        return max_error


class PoseCurve:
    def __init__(self, bone_animation, compression_settings):
        self.position_curve = CurveFitter(bone_animation, compression_settings, curve_type=1)
        self.rotation_curve = CurveFitter(bone_animation, compression_settings, curve_type=2)
        self.scale_curve = CurveFitter(bone_animation, compression_settings, curve_type=3)
        self.scale_rotation_curve = CurveFitter(bone_animation, compression_settings, curve_type=4)
        self.flip_curve = CurveFitter(bone_animation, compression_settings, curve_type=5)


class CompressionSettings:
    def __init__(self):
        self.linear_max_error = 0.0001
        """:type : float"""

        self.quat_max_error = 0.01
        """:type : float"""

        self.sampling_period = 0.01666667
        """:type : float"""

        self.frame_rate = 30.0
        """:type : float"""


class CompressedAnimation:
    def __init__(self, d, animations, a):
        anim = animations[a]
        """:type : Animation"""
        bone_animations = anim.bone_animations
        
        compression_settings = CompressionSettings()
        compression_settings.frame_rate = anim.frame_rate * anim.speed
        compression_settings.sampling_period = 1.0 / compression_settings.frame_rate / 2.0
        compression_settings.linear_max_error = d.linear_max_error
        compression_settings.quat_max_error = d.quat_max_error
        
        self.pose_curves = [None] * len(bone_animations)
        """:type : list[PoseCurve]"""
        
        for b in range(0, len(bone_animations)):  # for every bone...
            self.pose_curves[b] = PoseCurve(bone_animations[b], compression_settings)


class Animation:
    def __init__(self):
        self.source_name = None
        """:type : str"""

        self.relative_to = None
        """:type : str"""

        self.flags = None
        """:type : int"""

        self.frame_rate = 30.0
        """:type : float"""

        self.bone_animations = None
        """:type : list[BoneAnimation]"""

        self.start_frame = 0
        """:type : int"""

        self.end_frame = 0
        """:type : int"""

        self.speed = 1.0
        """:type : float"""

        self.frame_tags = []
        """:type : list[FrameTag]"""

        self.compressed_animation = None
        """:type : CompressedAnimation"""
        
        self.anim_bounds = None
        """:type : BoundBox"""

    def is_equivalent_to(self, other):
        return (self.source_name == other.source_name and
                self.relative_to == other.relative_to and
                self.start_frame == other.start_frame and
                self.end_frame == other.end_frame and
                self.speed == other.speed and
                self.flags == other.flags)
    
    def get_length(self):
        return self.end_frame - self.start_frame + 1


class AnimationNode:
    def __init__(self):
        self.name = None
        """:type : str"""

        self.flags = 0
        """:type : int"""

        self.animation = -1
        """:type : int"""

        self.param_name = None
        """:type : str"""

        self.min_value = None
        """:type : float"""

        self.max_value = None
        """:type : float"""

        self.blend_animations = None
        """:type : list[int]"""

        self.layer_animations = None
        """:type : list[int]"""

        self.index = -1
        """:type : int"""

    def debug_print(self, tabs=0):
        s = ('\t' * tabs) + 'name            = ' + str(self.name) + '\n'
        s += ('\t' * tabs) + 'flags           = ' + str(self.flags) + '\n'
        s += ('\t' * tabs) + 'animation       = ' + str(self.animation) + '\n'
        s += ('\t' * tabs) + 'minValue        = ' + str(self.min_value) + '\n'
        s += ('\t' * tabs) + 'maxValue        = ' + str(self.max_value) + '\n'
        s += ('\t' * tabs) + 'blendAnimations = ' + str(self.blend_animations) + '\n'
        s += ('\t' * tabs) + 'layerAnimations = ' + str(self.layer_animations) + '\n'
        return s

    def get_length(self, d):
        """
        :type d : ModelData
        """
        if self.animation < 0:
            return 0.0
        # returns the length of the animation_node, in seconds
        anim = d.animations[self.animation]
        num_frames = anim.end_frame - anim.start_frame - 1
        return num_frames / (anim.frame_rate * anim.speed)


class Sequence:
    def __init__(self):
        self.name = None
        """:type : str"""

        self.animation_node = -1
        """:type : int"""

        self.length = 0.0
        """:type : float"""
    
    def debug_print(self, tabs=0):
        s = ('\t' * tabs) + 'name            = ' + str(self.name) + '\n'
        s += ('\t' * tabs) + 'animationNode   = ' + str(self.animation_node) + '\n'
        s += ('\t' * tabs) + 'length          = ' + str(self.length) + '\n'
        return s


class BoneAnimation:
    def __init__(self):
        self.bone_index = -1
        """:type : int"""

        self.keys = []
        """:type : list[AffineParts]"""

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        return self.keys[item]

    def __setitem__(self, key, value):
        self.keys[key] = value


class AffineParts:
    def __init__(self, *args):
        self.translation = None
        """:type : Vec3"""

        self.rotation = None
        """:type : Quat"""

        self.scale = None
        """:type : Vec3"""

        self.scale_rotation = None
        """:type : Quat"""

        self.flip = 0.0
        """:type : float"""

        if len(args) == 1:
            if isinstance(args[0], AffineParts):  # make a copy of this affine parts
                a = args[0]
                self.translation = Vec3(a.translation)
                self.rotation = Quat(a.rotation)
                self.scale = Vec3(a.scale)
                self.scale_rotation = Quat(a.scale_rotation)
                self.flip = a.flip


def snuggle(q, k):
    sqrt_half = 0.7071067811865475244
    
    p = Quat()
    ka = [0.0, 0.0, 0.0]
    turn = -1
    
    ka[0] = k.x
    ka[1] = k.y
    ka[2] = k.z
    
    if ka[0] == ka[1]:
        if ka[0] == ka[2]:
            turn = 3
        else:
            turn = 2
    else:
        if ka[0] == ka[2]:
            turn = 1
        elif ka[1] == ka[2]:
            turn = 0
    
    if turn >= 0:
        neg = [False, False, False]
        
        qxtoz = Quat(); qxtoz.wxyz = [sqrt_half, 0.0, sqrt_half, 0.0]
        qytoz = Quat(); qytoz.wxyz = [sqrt_half, sqrt_half, 0.0, 0.0]
        qppmm = Quat(); qppmm.wxyz = [-0.5, 0.5, 0.5, -0.5]
        qpppp = Quat(); qpppp.wxyz = [0.5, 0.5, 0.5, 0.5]
        qmpmm = Quat(); qmpmm.wxyz = [-0.5, -0.5, 0.5, -0.5]
        qpppm = Quat(); qpppm.wxyz = [-0.5, 0.5, 0.5, 0.5]
        q0001 = Quat(); q0001.wxyz = [1.0, 0.0, 0.0, 0.0]
        q1000 = Quat(); q1000.wxyz = [0.0, 1.0, 0.0, 0.0]
        
        if turn == 0:
            qtoz = qxtoz
            q = q * qtoz
            ka[0], ka[2] = ka[2], ka[0]
        elif turn == 1:
            qtoz = qytoz
            q = q * qtoz
            ka[1], ka[2] = ka[2], ka[1]
        elif turn == 2:
            qtoz = q0001
        else:
            return q.get_conjugate(), k
        
        q = q.get_conjugate()
        
        mag = [0.0, 0.0, 0.0]
        
        mag[0] = q.z * q.z + q.w * q.w - 0.5
        mag[1] = q.x * q.z - q.y * q.w
        mag[2] = q.y * q.z + q.x * q.w
        
        for i in range(0, 3):
            neg[i] = (mag[i] < 0.0)
            if neg[i]:
                mag[i] = -mag[i]
        
        if mag[0] > mag[1]:
            if mag[0] > mag[2]:
                win = 0
            else:
                win = 2
        else:
            if mag[1] > mag[2]:
                win = 1
            else:
                win = 2
        
        if win == 0:
            p = q1000 if neg[0] else q0001
        elif win == 1:
            p = qppmm if neg[1] else qpppp
            ka[0], ka[1], ka[2] = ka[2], ka[0], ka[1]  # cycle backwards 1
        elif win == 2:
            if neg[2]:
                p = qmpmm
            else:
                p = qpppm
            ka[0], ka[1], ka[2] = ka[1], ka[2], ka[0]  # cycle forwards 1
        
        qp = q * p
        
        t = math.sqrt(mag[win] + 0.5)
        
        temp = Quat()
        temp.x = 0.0
        temp.y = 0.0
        temp.z = -qp.z / t
        temp.w = qp.w / t
        p = p * temp
        p = qtoz * p.get_conjugate()
    
    else:
        pa = [0.0, 0.0, 0.0, 0.0]
        neg = [False, False, False, False]
        par = False
        
        qa = [q[(i + 1) % 4] for i in range(0, 4)]
        
        for i in range(0, 4):
            pa[i] = 0.0
            neg[i] = qa[i] < 0.0
            if neg[i]:
                qa[i] = -qa[i]
            par ^= neg[i]

        # Find two largest components, indices in hi and lo
        if qa[0] > qa[1]:
            lo = 0
        else:
            lo = 1
            
        if qa[2] > qa[3]:
            hi = 2
        else:
            hi = 3
        
        if qa[lo] > qa[hi]:
            if qa[lo ^ 1] > qa[hi]:
                hi = lo; lo ^= 1
            else:
                hi, lo = lo, hi
        else:
            if qa[hi ^ 1] > qa[lo]:
                lo = hi ^ 1
        q_all = (qa[0] + qa[1] + qa[2] + qa[3]) * 0.5
        q_two = (qa[hi] + qa[lo]) * sqrt_half
        q_big = qa[hi]
            
        if q_all > q_two:
            if q_all > q_big:
                # all
                for i in range(0, 4):
                    pa[i] = -0.5 if neg[i] else 0.5
                if par == 0:
                    ka[0], ka[1], ka[2] = ka[2], ka[0], ka[1]  # cycle backwards 1
                else:
                    ka[0], ka[1], ka[2] = ka[1], ka[2], ka[0]  # cycle forwards 1
            else:
                # big
                pa[hi] = -1.0 if neg[hi] else 1.0
        else:
            if q_two > q_big:
                pa[hi] = -sqrt_half if neg[hi] else sqrt_half
                pa[lo] = -sqrt_half if neg[lo] else sqrt_half
                if lo > hi:
                    lo, hi = hi, lo
                if hi == 3:
                    hi = (lo + 1) % 3
                    lo = 3 - hi - lo
                ka[hi], ka[lo] = ka[lo], ka[hi]  # swap
            else:
                # big
                pa[hi] = -1.0 if neg[hi] else 1.0
            
        p.x = -pa[0]
        p.y = -pa[1]
        p.z = -pa[2]
        p.w = pa[3]
        
    k.x = ka[0]
    k.y = ka[1]
    k.z = ka[2]
        
    return p, k


def decompose_affine(matrix):
    parts = AffineParts()
    v = Vec3(); v[0] = matrix[0][3]; v[1] = matrix[1][3]; v[2] = matrix[2][3]
    parts.translation = v
    m = matrix.get_sub_matrix(3, 3)
    q, s, det = m.get_polar_decompose()
    
    if det < 0.0:
        q = -q
        parts.flip = -1.0
    else:
        parts.flip = 1.0
    
    parts.rotation = Quat()
    parts.rotation.set_from_matrix(q)
    
    parts.scale, U = s.get_spectral_decompose()
    
    parts.scale_rotation = Quat()
    parts.scale_rotation.set_from_matrix(U)
    
    p, parts.scale = snuggle(parts.scale_rotation, parts.scale)
    
    parts.scale_rotation = parts.scale_rotation * p
    return parts
    
    
def get_bone_index_by_name(d, name):
    for i in range(0, len(d.bones)):
        if d.bones[i].name == name:
            return i
    return -1


def locate_armature_in_scene(scene, arm_obj):
    """
    :type scene : bpy.types.Scene
    :type arm_obj : bpy_types.Object
    """
    for obj in scene.objects:
        if obj.data == arm_obj.data:
            return obj
    return None


def load_animations(d):
    """
    :type d: ModelData
    """
    m = d.model
    """:type : SparkModel"""
    bones = m.bones
    arm_obj = m.armature_object  # armature OBJECT
    anims = d.animations

    # Create a set of animations to sample.  The 'anims' list can contain duplicates, we don't need to sample them
    # more than once.
    anim_names = set()
    for a in anims:
        anim_names.add(a.source_name)
    anim_names = list(anim_names)
    # noinspection PyUnusedLocal
    raw_anim_tags = [[] for i in range(len(anim_names))]
    """:type : list[list[FrameTag]]"""
    
    # sample every bone for every animation
    raw_anims = [None] * len(anim_names)  # list of animations: each item is a list of bones
    """:type : list[list[list[Mat4]]]"""
    anim_frame_rates = [30.0] * len(anim_names)
    anim_start_frames = [0] * len(anim_names)
    """:type : list[int]"""
    anim_end_frames = [0] * len(anim_names)
    """:type : list[int]"""
    for a in range(0, len(raw_anims)):
        bone_anims = [None] * len(bones)  # list of bones: each item is a list of keys
        """:type : list[list[Mat4]]"""
        raw_anims[a] = bone_anims

        # change the context scene to the specified scene for this animation
        scene_index = bpy.data.scenes.find(anim_names[a])
        if scene_index == -1:
            raise SparkException("Scene '" + anim_names[a] + "' doesn't exist!")
        scene = bpy.data.scenes[scene_index]
        bpy.context.screen.scene = scene

        start_frame = math.floor(scene.frame_start)
        end_frame = math.ceil(scene.frame_end)
        num_keys = (end_frame - start_frame) + 1
        anim_start_frames[a] = start_frame  # need to keep this for when we trim animations
        anim_end_frames[a] = end_frame

        # create the frame tags
        markers = scene.timeline_markers
        for i in range(0, len(markers)):
            marker_frame = markers[i].frame
            marker_name = markers[i].name
            if end_frame >= marker_frame >= start_frame:
                new_tag = FrameTag()
                new_tag.frame = marker_frame
                new_tag.name = marker_name
                raw_anim_tags[a].append(new_tag)
        
        # resize the bone keys lists
        for i in range(0, len(bone_anims)):
            keys = [None] * num_keys
            """:type : list[Mat4]"""
            bone_anims[i] = keys
        
        # sample animation (all keys are in world-space at this stage in the sampling)
        for f in range(0, num_keys):
            bpy.context.scene.frame_set(f + start_frame)  # Do this several times to wiggle it into place, in case of
            bpy.context.scene.frame_set(f + start_frame)  # cyclical dependency problems.
            bpy.context.scene.frame_set(f + start_frame)
            bpy.context.scene.frame_set(f + start_frame)
            animation_arm_obj = locate_armature_in_scene(bpy.data.scenes[scene_index], arm_obj)
            if not animation_arm_obj:
                raise SparkException("Cannot locate matching armature for animation '" + anim_names[a] + "'.  Ensure "
                                     "there is an armature object in this scene, and ensure that its DATA references "
                                     "the same ARMATURE (not object, DATA).  If you created the scene via the 'Full "
                                     "Copy' method, the armature object's data will not match -- it will be set to a "
                                     "copy.")

            arm_world = Mat4(); arm_world.from_blender(animation_arm_obj.matrix_world)  # NOTE: a side effect of getting
            # the world-space matrix here -- instead of one-time in the visual scene -- is that the user can animate the
            # armature OBJECT itself, and that animation will be reflected in the game.  If that's a good thing or not
            # remains to be seen.

            if d.alternate_origin_object:  # transform armature by alternate origin, if necessary
                arm_world = Mat4(d.alternate_origin_object.matrix_world.inverted()) * arm_world
            
            for b in range(0, len(bone_anims)):
                pose_bone = animation_arm_obj.pose.bones[bones[b].name]
                if pose_bone is None:
                    raise SparkException("Something went wrong locating pose_bone from armature_bone, "
                                         "let there be panic!")
                key = Mat4()
                key.from_blender(pose_bone.matrix)
                key = arm_world * key
                key.fix_axes(reverse=True)  # perform reversed blender -> spark axes swap
                bone_anims[b][f] = key
        
    # scale the animation if needed
    s = d.scale_value
    if s < 0.0:
        raise SparkException("Negative scale values are not supported.  Aborting")
    if s == 0.0:
        raise SparkException("Zero-scale factor is not allowed.  Aborting")
        
    if s != 1.0:
        for a in range(0, len(raw_anims)):
            for b in range(0, len(raw_anims[a])):
                for f in range(0, len(raw_anims[a][b])):
                    key = raw_anims[a][b][f]
                    key.m03 *= s
                    key.m13 *= s
                    key.m23 *= s
    
    # Change zero scale to epsilon
    epsilon = sys.float_info.epsilon
    for a in range(0, len(raw_anims)):
        for b in range(0, len(raw_anims[a])):
            for f in range(0, len(raw_anims[a][b])):
                key = raw_anims[a][b][f]
                for c in range(0, 3):
                    if (key(0, c) * key(0, c) + key(1, c) * key(1, c) + key(2, c) * key(2, c) + key(3, c) * key(3, c))\
                            < epsilon:
                        key[c][c] = epsilon
                    
    # Compute the bounding boxes of every sampled animation
    mx = MinMaxVec3()
    for a in range(0, len(raw_anims)):
        for b in range(0, len(raw_anims[a])):
            verts = m.bone_bounds[b].get_verts()
            # transform the bone bounding verts to world space (they're stored in local space)
            for v in verts:
                for f in range(0, len(raw_anims[a][b])):
                    key = raw_anims[a][b][f]
                    transformed_vert = key * Vec3(v)
                    mx.min_max(transformed_vert)  # transform bound verts to every frame, adding them to the min-max
    d.model.bound_box.merge(mx)

    # Transform all the keys from global to local space
    # Loop through bones in reverse, since parents are always before children.  We need to transform children first.
    for a in range(0, len(raw_anims)):
        for b in range(len(raw_anims[a]) - 1, -1, -1):
            parent = bones[b].parent
            if parent is None:  # skip, it's already transformed properly
                continue
            p = bones.index(parent)  # index of parent bone
            if p > b:  # parent index is greater than child bone index
                raise Exception("Parent bone index > child bone index.  This should never happen!!!")
            for f in range(0, len(raw_anims[a][b])):
                key = raw_anims[a][b][f]
                p_key = raw_anims[a][p][f]
                key = p_key.get_inverse() * key
                raw_anims[a][b][f] = key

    # Convert matrices to affine parts
    affine_anims = [None] * len(raw_anims)
    """:type : list[list[list[AffineParts]]]"""
    for a in range(0, len(raw_anims)):
        raw_bone_anims = raw_anims[a]
        bone_anims = [None] * len(raw_bone_anims)
        """:type : list[list[AffineParts]]"""
        for b in range(0, len(raw_bone_anims)):
            raw_keys = raw_bone_anims[b]
            keys = [None] * len(raw_keys)
            """:type : list[AffineParts]"""
            for k in range(0, len(raw_keys)):
                mat = raw_keys[k]
                key = decompose_affine(mat)
                keys[k] = key
            bone_anims[b] = keys
        affine_anims[a] = bone_anims

    # Create individual animations from the master sampled animations list (what we just sampled)
    for a in range(0, len(anims)):
        master_anim = None  # full sampled animation that this sub-animation is a section of
        anim = anims[a]
        """:type : Animation"""
        for i in range(0, len(anim_names)):
            if anim_names[i] == anim.source_name:
                master_anim = i
                break
        master_start = anim_start_frames[master_anim]
        master_end = anim_end_frames[master_anim]

        scene = bpy.data.scenes[anim_names[master_anim]]

        if isinstance(anim.start_frame, str):
            if scene.timeline_markers.find(anim.start_frame) == -1:
                # noinspection PyTypeChecker
                raise SparkException("There is no timeline_marker named '" + anim.start_frame +
                                     "' in scene '" + scene.name + "'.")
            else:
                anim_start = scene.timeline_markers[anim.start_frame].frame
                anim.start_frame = anim_start  # replace string-value with marker's frame
        else:
            anim_start = math.floor(anim.start_frame)

        if isinstance(anim.end_frame, str):
            if scene.timeline_markers.find(anim.end_frame) == -1:
                # noinspection PyTypeChecker
                raise SparkException("There is no timeline_marker named '" + anim.end_frame +
                                     "' in scene '" + scene.name + "'.")
            else:
                anim_end = scene.timeline_markers[anim.end_frame].frame
                anim.end_frame = anim_end  # replace string-value with marker's frame
        else:
            anim_end = math.ceil(anim.end_frame)
        
        if anim_start:
            anim_start = master_start
        
        if anim_end:
            anim_end = master_end
        
        if anim_end > master_end:
            print("Warning: frame range of animation '", anim.source_name, "' is beyond the frame range of the scene.  "
                                                                           "Trimming and proceeding.")
            anim_end = master_end
        
        if anim_start < master_start:
            print("Warning: frame range of animation '", anim.source_name, "' is beyond the frame range of the scene.  "
                                                                           "Trimming and proceeding.")
            anim_start = master_start

        anim.frame_rate = anim_frame_rates[master_anim]
        
        index_start = anim_start - master_start
        index_end = anim_end - anim_start + index_start
        
        if index_start < 0:
            raise SparkException("Invalid frame range provided for animation '" + anim.source_name + "'.")
        if index_end < index_start:
            raise SparkException("Invalid frame range provided for animation '" + anim.source_name + "'.")

        # copy over relevant frame tags
        # frame tags' frame values are relative to the start of the animation, not the entire timeline
        # eg. frame tag @ frame 12 in an animation that starts @ frame 5 will have a frame value of 7
        for i in range(len(raw_anim_tags[master_anim])):
            if anim_start >= raw_anim_tags[a][i].frame >= anim_end:  # if the frame tag is in the animation range
                new_tag = FrameTag(raw_anim_tags[a][i])
                new_tag.frame -= master_start - anim_start
                anim.frame_tags.append(new_tag)

        bone_anims = [None] * len(affine_anims[master_anim])
        anim.bone_animations = bone_anims
        relative = False
        r_self = False
        if anim.flags & ANIMATION_FLAG_RELATIVE():  # if animation is 'relative' or 'relative_to'
            relative = True
            if anim.relative_to is None:
                r_self = True
        for b in range(0, len(bone_anims)):  # for every bone
            bone_len = index_end - index_start + 1
            bone_keys = [None] * bone_len
            new_bone_anim = BoneAnimation()
            new_bone_anim.bone_index = b
            new_bone_anim.keys = bone_keys
            bone_anims[b] = new_bone_anim
            
            if relative:
                if r_self:
                    base_key = affine_anims[master_anim][b][0]  # relative_to_start makes entire animation
                                                                # relative to first key
                else:
                    raise SparkException("'relative_to' is not supported.")
                
                for k in range(0, bone_len):
                    src_key = affine_anims[master_anim][b][index_start + k]
                    new_affine = AffineParts()
                    new_affine.translation = src_key.translation - base_key.translation
                    new_affine.rotation = base_key.rotation.get_conjugate() * src_key.rotation
                    # divide each component of src by corresponding base component
                    new_affine.scale = Vec3([src_key.scale[s] / base_key.scale[s] for s in range(0, 3)])
                    new_affine.scale_rotation = base_key.scale_rotation.get_conjugate() * src_key.scale_rotation
                    new_affine.flip = src_key.flip / base_key.flip
                    bone_keys[k] = new_affine
            
            else:  # not relative, can just get a slice of the keys
                new_bone_anim.keys = affine_anims[master_anim][b][index_start:index_end + 1]
    
    # Create compressed versions of the animations
    if d.compression_enabled:
        for a in range(0, len(anims)):
            if anims[a].get_length() <= 2:
                # can't compress an animation that's 2 or fewer frames in length, skipping this one
                continue
            anims[a].compressed_animation = CompressedAnimation(d, anims, a)

    for i in range(0, len(d.sequences)):
        seq = d.sequences[i]
        anim_node = d.animation_nodes[seq.animation_node]
        seq.length = anim_node.get_length(d)
