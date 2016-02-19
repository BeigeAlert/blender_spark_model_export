# 1455664595
# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

# Mesh-related stuff goes here

from spark_common import *

# Constants
def MAX_BONES_PER_VERT(): return 4  # used in leiu of actual constants in python :(


class HeadVertex:
    def __init__(self):
        self.head = None
        """:type : Vertex"""
        self.children = []
        """:type : list[Vertex]"""
        self.original_vertex_index = -1
        """:type : int"""

    def get_verts(self):
        vert_list = [self.head]
        if self.children:
            vert_list.extend(self.children)
        return vert_list

    def get_vert_from_loop_index(self, loop_index):
        if self.head.original_loop_index == loop_index:
            return self.head
        else:
            for c in self.children:
                if c.original_loop_index == loop_index:
                    return c
        raise SparkException("Triangle requested loop index that doesn't exist for the corresponding vertex!")


class Vertex:
    def __init__(self):
        self.co = None
        self.nrm = None
        self.tan = None
        self.bin = None
        self.t_co = None
        
        self.bone_weights = []
        """:type : list[BoneWeight]"""
        self.color = [1, 1, 1, 1]  # pretty sure this is unused
        
        # used for exporting from blender only
        self.original_loop_index = -1 # blender loop-index (per object, not global to combined export mesh)
        self.smooth_influence = 0.0
        self.triangles = []
        """:type : list[Triangle]"""
        self.head = None
        """:type : HeadVertex"""

        self.written_index = -1

    def __eq__(self, other):
        """
        :type other: Vertex
        """
        dist = lambda a,b: abs(a-b)
        if dist(self.co[0], other.co[0]) > 0.00001: return False
        if dist(self.co[1], other.co[1]) > 0.00001: return False
        if dist(self.co[2], other.co[2]) > 0.00001: return False
        if dist(self.nrm[0], other.nrm[0]) > 0.00001: return False
        if dist(self.nrm[1], other.nrm[1]) > 0.00001: return False
        if dist(self.nrm[2], other.nrm[2]) > 0.00001: return False
        if dist(self.tan[0], other.tan[0]) > 0.00001: return False
        if dist(self.tan[1], other.tan[1]) > 0.00001: return False
        if dist(self.tan[2], other.tan[2]) > 0.00001: return False
        if dist(self.bin[0], other.bin[0]) > 0.00001: return False
        if dist(self.bin[1], other.bin[1]) > 0.00001: return False
        if dist(self.bin[2], other.bin[2]) > 0.00001: return False
        if dist(self.t_co[0], other.t_co[0]) > 0.00001: return False
        if dist(self.t_co[1], other.t_co[1]) > 0.00001: return False
        return True

    # Add a bone and weight to the list for this vertex.  If it's already at capacity,
    # replace the lightest of the bone weights with this new one, provided the new one
    # is heavier.
    def add_bone_weight(self, bone_index, bone_weight):
        if len(self.bone_weights) >= MAX_BONES_PER_VERT():
            lightest = 0
            for i in range(1, len(self.bone_weights)):
                if self.bone_weights[i].weight < self.bone_weights[lightest].weight:
                    lightest = i
            if self.bone_weights[lightest].weight < bone_weight:
                self.bone_weights[lightest].weight = bone_weight
                self.bone_weights[lightest].index = bone_index
        else:  # some room available
            bw = BoneWeight()
            bw.index = bone_index
            bw.weight = bone_weight
            self.bone_weights.append(bw)

    # bone weights need to add up to 1.0
    def normalize_bone_weights(self):
        total = 0.0
        for i in range(0, len(self.bone_weights)):
            total += self.bone_weights[i].weight
        
        for i in range(0, len(self.bone_weights)):
            self.bone_weights[i].weight = self.bone_weights[i].weight / total

    def get_bone_list(self, bone_offset=0):
        b_list = []
        for i in range(len(self.bone_weights)):
            if self.bone_weights[i].weight >= 0.00001:
                b_list.append(self.bone_weights[i].index + bone_offset)
        return b_list
        

class Triangle:
    def __init__(self):
        self.verts = []
        """:type : list[Vertex]"""
        self.normal = (0.0, 0.0, 0.0)
        self.material = -1


# not used in lieu of blender bones, but used for any extra bones added by the exporter.  Currently, bones are only
# ever added to give cameras a parent.
class Bone:
    def __init__(self):
        self.name = ''
        """:type : str"""

        self.bone_to_world_matrix = None  # These bones will always be at the top level, eg no bone_to_parent matrix.
        """:type : Mat4"""


class BoneWeight:
    def __init__(self):
        self.index = 0
        self.weight = 0.0


class BoneNode:
    def __init__(self):
        self.child = -1
        self.sibling = -1


class Material:
    def __init__(self):
        self.blender_material = None
        self.spark_material = ''
    
    def blender_mats_equal(self, other):
        if self.blender_material == other.blender_material:
            return True
        return False
    
    def spark_mats_equal(self, other):
        if self.spark_material == other.spark_material:
            return True
        return False


class SparkModel:
    def __init__(self):
        self.verts = []
        """:type : list[Vertex]"""

        self.triangles = []
        """:type : list[Triangle]"""

        self.bones = []  # bpy bone objects
        """:type : list[bpy_types.Bone]"""

        self.extra_bones = []  # Bone objects
        """:type : list[Bone]"""

        self.bone_bounds = []  # bone bounding boxes
        """:type : list[BoundBox]"""

        self.bone_base_poses = []  # bone's bone to parent transform
        """:type : list[Mat4]"""

        self.bone_to_index = {}
        """:type : dict"""

        self.bone_world_mats = []  # bone's bone to world transform
        """:type : list[Mat4]"""

        self.materials = []  # Material objects
        """:type : list[Material]"""

        self.armature_object = None
        """:type : bpy_types.Object"""

        self.bound_box = MinMaxVec3()
        """:type : MinMaxVec3"""
     
    def find_blender_material(self, material):
        for i in range(0, len(self.materials)):
            if self.materials[i].blender_material == material:
                return i
        return -1
    
    def find_spark_material(self, s_mat):  # returns the first index that matches.
        for i in range(0, len(self.materials)):
            if self.materials[i].spark_material == s_mat:
                return i
        return -1


