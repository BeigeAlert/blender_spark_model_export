# 1453703568
# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

import bpy
import math
from model_compile_parser import *
from spark_model import *
from spark_animation import *
from spark_physics import *
from spark_common import *
from spark_writer import *
import sys

# Constants
def MAX_BONES_PER_FACE_SET(): return 60
def DEV_MATERIAL(): return 'materials/dev/checkerboard.material'
def INCHESPERMETER(): return 39.3700787

def GetFileName():
    return (bpy.data.filepath.replace('\\','/').split('/'))[-1]

blend_name = GetFileName()

class FaceSet:
    def __init__(self):
        self.material_index = -1
        """:type : int"""

        self.faces = []
        """:type : list[int]"""

        self.bones = []
        """:type : list[int]"""


class AttachPoint:
    def __init__(self):
        self.name = ''
        """:type : str"""

        self.coords = None
        """:type : Coords"""

        self.parent_bone = None
        """:type : bpy_types.Bone"""


class TangentPair:
    def __init__(self):
        self.tangent = Vec3(0, 0, 0)
        self.binormal = Vec3(0, 0, 0)


class Camera:
    def __init__(self):
        self.name = ''
        """:type : str"""

        self.parent_bone_blender = None  # parented to pre-existing bone in blender
        """:type : bpy_types.Bone"""

        self.parent_bone_extra = None  # extra bone created by exporter to hold camera (if camera is un-parented)
        """:type : int"""

        self.fov = 1.570796327  # default horizontal FOV of 90 degrees ( pi/2 )
        """:type : float"""

        self.coords = None
        """:type : Coords"""


class ModelData:
    def __init__(self):
        self.animations = []
        """:type : list[Animation]"""

        self.sequences = []
        """:type : list[Sequence]"""

        self.animation_nodes = []
        """:type : list[AnimationNode]"""
        
        self.scale_value = 1.0
        """:type : float"""

        self.compression_enabled = True
        """:type : bool"""

        self.linear_max_error = None
        """:type : float"""

        self.quat_max_error = None
        """:type : float"""

        self.joints = []
        """:type : list[Joint]"""

        self.solids = []
        """:type : list[Solid]"""

        self.collision_pairs = []
        """:type : list[CollisionPair]"""

        self.read_collision_pairs = []

        self.attach_points = []  # just the names, no objects just yet

        self.attach_point_objects = []  # the actual objects
        """:type : list[AttachPoint]"""

        self.animation_model = None
        """:type : str"""

        self.geometry_group = None
        """:type : list[str]"""

        self.physics_groups = None
        """:type : list[list[str]]"""
        
        self.collision_reps = []
        """:type : list[CollisionRep]"""

        self.collision_rep_entries = []
        """:type : list[CollisionRepEntry]"""

        self.model = None
        """:type : SparkModel"""

        self.disable_alternate_origin = False
        """:type : bool"""

        self.alternate_origin_object = None
        """:type : bpy_types.Object"""

        self.cameras = []
        """:type : list[Camera]"""

        self.face_sets = []
        """:type : list[FaceSet]"""

        self.blend_parameters = []
        """:type : list[str]"""

        self.add_world_bone = False  # Directive to create an extra bone to allow for static geometry in addition to
                                     # animated geometry
        

def add_dummy_material(m):
    index = m.find_blender_material(None)
    if index != -1:
        return index
    new_material = Material()
    new_material.blender_material = None
    new_material.spark_material = DEV_MATERIAL()
    m.materials.append(new_material)
    return len(m.materials) - 1


def add_material(m, material):
    index = m.find_blender_material(material)
    if index != -1:
        return index
    
    new_material = Material()
    new_material.blender_material = material
    m.materials.append(new_material)
    
    # check to see if the user has explicitly defined a spark material path
    explicit = material.get('SparkMaterial')
    if (explicit):
        new_material.spark_material = explicit
        return len(m.materials) - 1
    
    best_candidate = None
    for slot in material.texture_slots:
        if best_candidate is None and slot.texture is not None:
            best_candidate = slot.texture
        if slot.use_map_color_diffuse and slot.texture is not None:
            best_candidate = slot.texture
            break
    
    if best_candidate.type != 'IMAGE':
        new_material.spark_material = DEV_MATERIAL()
        return len(m.materials) - 1
    
    img = best_candidate.image
    if img.source != 'FILE':
        new_material.spark_material = DEV_MATERIAL()
        return len(m.materials) - 1
    
    if img.filepath == '':
        new_material.spark_material = DEV_MATERIAL()
        print(blend_name, ": Invalid texture path. (NOTE: packing textures into the .blend is not supported... "
              "or wise even if it WAS... ;) )")
        return len(m.materials) - 1
    
    path = img.filepath
    path = path.replace('modelsrc', 'models')
    path = path.replace('materialsrc', 'materials')
    path = path.replace('\\', '/')
    path = path.split('/')
    path = [p for p in path if p.replace('.', '') != '']
    for i in range(len(path) - 1, -1, -1):  # loop backwards to find the last occurrence of either of these names
        if path[i] == 'models' or path[i] == 'materials':
            path = '/'.join(path[i:])
            break
    path = '.'.join(path.split('.')[:-1]) + '.material'  # chop off extension, and append '.material'
    new_material.spark_material = path
    return len(m.materials) - 1


def sum_vectors(v1, v2):
    if len(v1) != len(v2):
        raise SparkException("Cannot sum vectors of differing sizes!")
    for i in range(0, len(v1)):
        v1[i] += v2[i]
    return v1


def negate_vector(vect):
    new_vect = [None] * len(vect)
    for i in range(0, len(vect)):
        new_vect[i] = vect[i] * -1.0
    return new_vect


def dot_product(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def normalized_dot_product(vect1, vect2):
    v1 = normalize_vector(vect1)
    v2 = normalize_vector(vect2)
    return max(min(v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2], 1.0), -1.0)


def normalize_vector(vect):
    mag = 0.0
    for d in vect:
        mag += d * d
    mag = math.sqrt(mag)
    if mag == 0.0:
        return 0.0, 0.0, 0.0
    result = []
    for d in vect:
        result.append(d / mag)
    return tuple(result)


def add_bone_to_list(i, bone_nodes, new_to_old):
    if i == -1:
        return
    
    new_to_old.append(i)
    add_bone_to_list(bone_nodes[i].sibling, bone_nodes, new_to_old)
    add_bone_to_list(bone_nodes[i].child, bone_nodes, new_to_old)


def pre_process_bones(d):
    # locate the armature in the geometry group to use.  Only one armature can be accepted.
    group_objs = [obj for obj in bpy.data.groups[d.geometry_group[0]].objects]
    scene_objs = [obj for obj in bpy.data.scenes[d.geometry_group[1]].objects]
    objs = [obj for obj in group_objs if obj in scene_objs]
    arm_obj = None
    for obj in objs:
        if not obj.type == 'ARMATURE':
            continue
        if arm_obj is not None:
            raise SparkException("More than one armature present in both geometry group and geometry scene.")
        else:
            arm_obj = obj

    if not arm_obj:
        if d.animations:  # animations isn't empty, but no armature present
            print(blend_name, ": Warning: animations declared, but there is no base-armature present in the visual scene.  "
                  "Proceeding as a static-mesh export.")
        return None
    bones = arm_obj.data.bones
    # calculate base-pose matrices
    if arm_obj:
        d.model.bone_base_poses = [None] * len(bones)
        d.model.bone_world_mats = [None] * len(bones)
        amat = Mat4(); amat.from_blender(arm_obj.matrix_world)
        if d.alternate_origin_object: # transform armature by alternate origin
            amat = Mat4(d.alternate_origin_object.matrix_world.inverted()) * amat

        for i in range(0, len(bones)):
            # while we're looping through bones, need to check to make sure none of them are called "world-space"
            # this is a reserved name.
            if bones[i].name.lower() == 'world-space':
                raise SparkException("Cannot name a bone 'world-space'.  This is reserved for the implicitly-created "
                                     "bone for static geometry.")

            bone_world = Mat4(); bone_world.from_blender(arm_obj.pose.bones[bones[i].name].matrix)
            bone_world = amat * bone_world
            d.model.bone_world_mats[i] = bone_world
            if bones[i].parent:  # bone has a parent
                parent_world = Mat4(); parent_world.from_blender(arm_obj.pose.bones[bones[i].name].parent.matrix)
                parent_world = amat * parent_world
                bone_local = parent_world.get_inverse() * bone_world
            else:
                bone_local = Mat4(bone_world)
                bone_local.fix_axes(reverse=True)  # perform reversed blender -> spark axes swap
                # to compensate for the mesh verts getting "fixed" as well.

            # scale translation by scale factor
            bone_local[0][3] *= d.scale_value
            bone_local[1][3] *= d.scale_value
            bone_local[2][3] *= d.scale_value

            d.model.bone_base_poses[i] = bone_local
            del bone_local

    # Create a mapping of the bones such that for every bone, all of its siblings are at a greater index than its
    # children.  To be sorted in the next step.
    # noinspection PyUnusedLocal
    bone_nodes = [BoneNode() for i in range(len(bones))]  # Indices of bone_nodes match up to indices of bones.
    bones_list = bones.values()  # for some damn reason using .find() always gives -1...  gotta search a list instead.

    # Create bone tree
    for i in range(0, len(bones)):
        if bones[i].parent is not None:
            parent_node = bone_nodes[bones_list.index(bones[i].parent)]
            if parent_node.child != -1:
                bone_nodes[i].sibling = parent_node.child
            parent_node.child = i
    
    # Create correspondence list
    new_to_old = []
    for i in range(0, len(bones)):
        if bones[i].parent is None:
            add_bone_to_list(i, bone_nodes, new_to_old)
    
    # Create mapping from old to new
    old_to_new = [None] * len(new_to_old)
    for i in range(0, len(new_to_old)):
        old_to_new[new_to_old[i]] = i
    
    # Reorder bones
    new_bones = [None] * len(bones)
    for i in range(0, len(bones)):
        new_bones[i] = bones[new_to_old[i]]
    
    d.model.bones = new_bones
    d.model.armature_object = arm_obj

    # Because we'll need to get an index from a bone's name, and we can't go modifying blender's internal bone class,
    # we'll use a dict.

    d.model.bone_to_index = {}
    for i in range(len(d.model.bones)):
        d.model.bone_to_index[d.model.bones[i].name] = i

    # Reorder bone base/world matrices to match new indexes
    old_base_poses = d.model.bone_base_poses
    old_world_mats = d.model.bone_world_mats

    new_base_poses = [None] * len(old_base_poses)
    new_world_mats = [None] * len(old_world_mats)
    for i in range(len(new_base_poses)):
        new_base_poses[i] = old_base_poses[new_to_old[i]]
        new_world_mats[i] = old_world_mats[new_to_old[i]]

    d.model.bone_base_poses = new_base_poses
    d.model.bone_world_mats = new_world_mats


def calculate_bone_bounding_boxes(d):
    # calculate the bound box of each bone's base pose, in the bone's local coordinates
    bones = d.model.bones
    
    # build influence list
    # noinspection PyUnusedLocal
    influences = [[] for i in range(0, len(bones))]  # one list per bone -- each bone's list is a list of vertices
    """:type : list[list[int]]"""
    verts = d.model.verts
    for i in range(0, len(verts)):
        for b in verts[i].bone_weights:
            influences[b.index].append(i)  # add this vertex index to this bone's list
    
    # for every bone, transform the vertices to bone space, and size the bound box appropriately
    d.model.bone_bounds = [None] * len(bones)
    bounds = d.model.bone_bounds
    for i in range(0, len(bones)):
        bone_to_world = Mat4(d.model.bone_world_mats[i])  # get a copy of it

        # scale matrix accordingly if child bone
        if d.scale_value != 1.0 and bones[i].parent:
            bone_to_world[0][3] *= d.scale_value
            bone_to_world[1][3] *= d.scale_value
            bone_to_world[2][3] *= d.scale_value

        world_to_bone = bone_to_world.get_inverse()

        v_list = influences[i]
        minmax = MinMaxVec3()
        for j in range(0, len(v_list)):
            co = Vec3(verts[v_list[j]].co, fix_axes=True)  # world-space coordinates
            b_co = world_to_bone * co  # transform from world-space coordinates, to bone-space local coordinates
            minmax.min_max(b_co)
        bounds[i] = BoundBox(minmax)


# Loads all the geometry specified in the model_compile text block
def load_geometry(d):
    d.model = SparkModel()

    result = bpy.data.groups.find(d.geometry_group[0])
    if result == -1:
        raise SparkException("Geometry group '" + d.geometry_group[0] + "' does not exist!")

    result = bpy.data.scenes.find(d.geometry_group[1])
    if result == -1:
        raise SparkException("Geometry scene '" + d.geometry_group[1] + "' does not exist!")

    # Little hack to work around bad transforms caused by cyclic redundancy issues.
    visual_scene = bpy.data.scenes[result]
    frame = visual_scene.frame_start
    bpy.context.screen.scene = visual_scene
    bpy.context.scene.frame_set(frame)  # Do this several times to ensure the bones are where they should be.  They can
    bpy.context.scene.frame_set(frame)  # be off if the user has created a cyclical dependency.
    bpy.context.scene.frame_set(frame)  # There... 4 resets ought to do it... if they've got more than 4 bones in a
    bpy.context.scene.frame_set(frame)  # cyclic dependency, well they can just deal with it...

    # Create a merged, sorted, indexed list of bones.  Also get the relative base-transforms for each bone.
    pre_process_bones(d)

    # intersection of group and scene objects
    group_objs = [obj for obj in bpy.data.groups[d.geometry_group[0]].objects]
    scene_objs = [obj for obj in bpy.data.scenes[d.geometry_group[1]].objects]
    objs = [obj for obj in group_objs if obj in scene_objs]
    found_mesh = False  # will be set True when a suitable mesh is found, for error reporting
    for obj in objs:
        isCurve = False
        if obj.type == 'CURVE':
            old_setting = obj.data.use_uv_as_generated
            obj.data.use_uv_as_generated = True #force uv generation on, otherwise we can't generate tangents
            isCurve = True
        elif not obj.type == 'MESH':
            continue

        found_mesh = True

        # triangulate mesh first
        scene = bpy.context.scene
        temp_object = bpy.data.objects.new('temp_processing_object', bpy.data.meshes.new_from_object(scene, obj,
                                                                                                     True, 'PREVIEW'))
        if isCurve:
            obj.data.use_uv_as_generated = old_setting #reset this back to whatever user had before.
        me = temp_object.data
        scene.objects.link(temp_object)
        scene.objects.active = temp_object
        # noinspection PyCallByClass,PyTypeChecker
        bpy.ops.object.mode_set(mode='EDIT', toggle=False)
        bpy.ops.mesh.reveal()
        # noinspection PyCallByClass,PyTypeChecker
        bpy.ops.mesh.select_all(action='SELECT')
        # noinspection PyCallByClass,PyTypeChecker,PyArgumentList
        bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
        # noinspection PyCallByClass
        bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
        
        if me.uv_layers.active:
            me.calc_tangents()  # only calculate tangents if UV data is present.

        me.transform(obj.matrix_world)

        # transform to alternate origin, if applicable
        if d.alternate_origin_object:
            me.transform(d.alternate_origin_object.matrix_world.inverted())
        
        # In Spark, vertices are split by triangle because the normal/tangent/bitangent vectors and
        # texture coordinates are stored one-per-vertex.  Not so in blender.  In Blender, a mesh
        # vertices are divided into face-corners, or "loops".  Therefore, a spark-model "vertex" is
        # really a Blender "loop".  To greatly improve performance, we calculate the smoothed normal
        # vector by blender-vertex -- not by blender-loop, as it will always be the same anyways.
        # We load up all of Blender's vertices into a list of HeadVertex objects.  The "HeadVertex"
        # is just a class that holds other Vertex classes: one being the "HeadVertex", the others a
        # list of children derived from the same vertex as blender-loops.

        head_verts = [None] * len(me.vertices)
        """:type : list[HeadVertex]"""
        
        num_loops = len(me.loops)  # useful to know later how many loops total, without having to count each list
        for i in range(0, len(me.loops)):
            new_vert = Vertex()
            new_vert.original_loop_index = i
            if not head_verts[me.loops[i].vertex_index]:  # this loop's parent vert has not been initialized yet
                new_head_vert = HeadVertex()
                new_head_vert.head = new_vert
                new_head_vert.original_vertex_index = me.loops[i].vertex_index
                new_vert.head = new_head_vert
                head_verts[new_head_vert.original_vertex_index] = new_head_vert
            else:
                head_verts[me.loops[i].vertex_index].children.append(new_vert)
                new_vert.head = head_verts[me.loops[i].vertex_index]
            coords = me.vertices[new_vert.head.original_vertex_index].co
            # Notice coords are cycled below... this is the blender -> spark axes cycle
            new_vert.co = [coords[1] * d.scale_value, coords[2] * d.scale_value, coords[0] * d.scale_value]
            coords = me.loops[i].normal
            new_vert.nrm = [coords[1], coords[2], coords[0]]
            coords = me.loops[i].tangent
            new_vert.tan = [coords[1], coords[2], coords[0]]
            coords = me.loops[i].bitangent
            new_vert.bin = [coords[1], coords[2], coords[0]]

            # add this vertex to the bounding box of the model
            d.model.bound_box.min_max(new_vert.co[:])

            try:
                t_coords = me.uv_layers.active.data[i].uv
            except AttributeError:
                t_coords = [0.0, 0.0]
            new_vert.t_co = [t_coords[0], 1.0 - t_coords[1]]

        tris = [None] * len(me.polygons)
        """:type : list[Triangle]"""
        for i in range(0, len(me.polygons)):
            new_tri = Triangle()
            try:
                new_tri.material = add_material(d.model,
                                                temp_object.material_slots[me.polygons[i].material_index].material)
            except IndexError:
                new_tri.material = add_dummy_material(d.model)
            new_tri.verts = [head_verts[me.loops[x].vertex_index].get_vert_from_loop_index(x)
                             for x in me.polygons[i].loop_indices]

            p1 = Vec3(new_tri.verts[0].co)
            p2 = Vec3(new_tri.verts[1].co)
            p3 = Vec3(new_tri.verts[2].co)
            
            # Assign the backwards-link to each vertex
            for j in range(3):
                new_tri.verts[j].triangles.append(new_tri)
            tris[i] = new_tri

        # Process bone weights
        arm_obj = d.model.armature_object
        if arm_obj is not None:
            for h in head_verts:
                found_group = False
                for g in obj.data.vertices[h.original_vertex_index].groups:
                    group_index = g.group
                    bone_name = obj.vertex_groups[group_index].name
                    try:
                        h.head.add_bone_weight(d.model.bone_to_index[bone_name], g.weight)
                        found_group = True
                    except KeyError:
                        # group wasn't a bone
                        continue
                if not found_group:
                    d.add_world_bone = True  # we located a vertex not mapped to any bone.  For it to survive, we must
                                             # map it to a new bone, which we'll make static.
                else:
                    h.head.normalize_bone_weights()
                    for l in h.children:  # propagate weights to the vert's other loops
                        l.bone_weights = h.head.bone_weights
        
        # Clean up temporary object
        scene.objects.unlink(temp_object)
        bpy.data.objects.remove(temp_object)
        bpy.data.meshes.remove(me)

        # Smooth out the tangent/bitangent vectors at each vertex with the other tangent/bitangent values
        # This means we average the tangent/bitangent values of each loop for every vertex.  A loop is only
        # averaged if, given a tan/bit basis to compare to, their dot products are > 0.  So this creates the
        # potential to have more than one averaged tan/bit basis per vertex.  So we'll keep re-running this
        # averaging procedure until we've averaged up every loop.

        # Merge duplicate loops.
        for h in head_verts:  # for every vertex, look in the loops for duplicates
            if not h:
                continue
            loops = h.get_verts()
            for i in range(0, len(loops) - 1):
                if not loops[i]:
                    continue
                for j in range(i + 1, len(loops)):
                    if not loops[j]:
                        continue
                    v1 = loops[i]
                    v2 = loops[j]
                    if v1 == v2:
                        # merge v2 into v1
                        # switch all triangle->vertex references over to the new vertex
                        for t in v2.triangles:
                            for v in range(len(t.verts)):
                                if t.verts[v] == v2:
                                    t.verts[v] = v1
                        v1.triangles = list(set(v1.triangles) | set(v2.triangles))
                        loops[j] = None

            # condense list: remove None entries.  Head vertex is always safe.  Loop over children.
            h.children = [c for c in h.children if c is not None]

        # append to the verts and triangles lists
        count = 0
        for h in head_verts:
            if not h:
                continue
            count += 1 + (len(h.children) if h.children else 0)
        verts = [None] * count
        """:type : list[Vertex]"""
        next_index = 0
        for h in head_verts:
            if not h:
                continue
            verts[next_index] = h.head
            next_index += 1
            if h.children:
                for c in h.children:
                    verts[next_index] = c
                    next_index += 1
        verts = verts[:next_index]
        d.model.verts.extend(verts)
        d.model.triangles.extend(tris)

    if not found_mesh:
        raise SparkException('No (valid) MESH objects found in scene "' + d.geometry_group[1] +
                             "' belonging to group '" + d.geometry_group[0] + "'.  Aborting.")

    # calculate the bone bounding boxes
    calculate_bone_bounding_boxes(d)

    # add cameras
    for obj in objs:
        if obj.type == 'CAMERA':
            new_camera = Camera()
            new_camera.name = obj.name
            new_camera.fov = obj.data.angle_x

            if obj.parent is None:
                no_parent = True
            elif obj.parent.type != 'ARMATURE':  # NOTE THE PERIOD!!!  (Parent's type)
                no_parent = True
            elif obj.parent_type != 'BONE':  # NOTE THE UNDERSCORE!!!  (type of relationship TO parent)
                no_parent = True
                print(blend_name, ": Warning!  Camera '", obj.name, "' parented to the armature OBJECT, not to an individual bone "
                      "within the armature.  Camera will be treated as static.", sep='')
            else:
                no_parent = False

            if no_parent:
                # If there's no parent, we need to transform to world-space, and create a new bone for this camera.
                new_bone = Bone()
                new_bone.name = obj.name

                new_bone.bone_to_world_matrix = Mat4(obj.matrix_world)
                if d.alternate_origin_object and not d.disable_alternate_origin:
                    new_bone.bone_to_world_matrix = Mat4(d.alternate_origin_object.matrix_world.inverted())\
                                                    * new_bone.bone_to_world_matrix
                new_bone.bone_to_world_matrix[0][3] *= d.scale_value
                new_bone.bone_to_world_matrix[1][3] *= d.scale_value
                new_bone.bone_to_world_matrix[2][3] *= d.scale_value

                new_bone.bone_to_world_matrix.fix_axes(reverse=True)

                new_camera.coords = Coords.get_identity()
                new_camera.parent_bone_extra = len(d.model.extra_bones)
                d.model.extra_bones.append(new_bone)
            else:
                # If there is a parent, we need to get the camera's coordinates in bone-space.
                matrix = d.model.bone_world_mats[d.model.bone_to_index[obj.parent_bone]].get_inverse() *\
                     Mat4(obj.matrix_world)

                # scale matrix translation components
                matrix[0][3] *= d.scale_value
                matrix[1][3] *= d.scale_value
                matrix[2][3] *= d.scale_value

                new_camera.coords = Coords(Mat4(matrix))
                new_camera.parent_bone_blender = obj.parent_bone

            new_camera.coords.x_axis *= -1.0  # Spark cameras point down +Z axis, but Blender cameras point down -Z
            new_camera.coords.z_axis *= -1.0  # axis.  Flip the Z component, then even it out by flipping X too.

            d.cameras.append(new_camera)

    # add attach points
    for i in range(0, len(d.attach_points)):
        obj = bpy.data.objects[d.attach_points[i]]  # we know this is a valid index from the parser stage
        if obj.type != 'EMPTY':
            print(blend_name, ": Warning!  object '", d.attach_points[i], "' is not an 'EMPTY'-type object.  This is unusual, but "
                  "workable.  Proceeding, but you really should fix this for clarity-sake.", sep='')
        if obj not in objs:
            print(blend_name, ": Warning!  AttachPoint '", d.attach_points[i], "' is not part of both the geometry group and scene.  "
                  "Skipping.", sep='')
            continue
        new_attach = AttachPoint()
        new_attach.name = obj.name

        if obj.parent is None:
            no_parent = True
        elif obj.parent.type != 'ARMATURE':  # NOTE THE PERIOD!!!  (Parent's type)
            no_parent = True
        elif obj.parent_type != 'BONE':  # NOTE THE UNDERSCORE!!!  (type of relationship TO parent)
            no_parent = True
            print(blend_name, ": Warning!  Attach point '", obj.name, "' parented to the armature OBJECT, not to an individual bone "
                  "within the armature.  Attach point will be treated as static.", sep='')
        else:
            no_parent = False

        if no_parent:
            # If there's no parent, we simply convert the world-space matrix to coords and be done with it.
            if d.alternate_origin_object:
                new_attach.bone_to_world_matrix = Mat4(d.alternate_origin_object.matrix_world.inverted()
                                                       * obj.matrix_world)
            else:
                new_attach.bone_to_world_matrix = Mat4(obj.matrix_world)
            new_attach.parent_bone = None
        else:
            matrix = d.model.bone_world_mats[d.model.bone_to_index[obj.parent_bone]].get_inverse() *\
                     Mat4(obj.matrix_world)

            # scale matrix translation components
            matrix[0][3] *= d.scale_value
            matrix[1][3] *= d.scale_value
            matrix[2][3] *= d.scale_value

            new_attach.coords = Coords(matrix)
            new_attach.parent_bone = obj.parent_bone
        d.attach_point_objects.append(new_attach)
    

def find_equivalent_animation(d, animation):
    for i in range(0, len(d.animations)):
        if animation.is_equivalent_to(d.animations[i]):
            return i
    return -1


def get_animation_node_by_name(d, name):
    for i in range(0, len(d.animation_nodes)):
        if d.animation_nodes[i].name == name:
            return i
    return -1


def read_float(t):
    s = t.get_token()
    try:
        x = float(s)
        return x
    except:
        raise SparkException("Expected numerical value at line " + str(t.get_line()) + ", got " + s + " instead.")


def read_animation_node(d, reader):
    """
    :type d: ModelData
    """
    anim_node = AnimationNode()
    anim_node.animation = -1
    anim_node.flags = 0
    
    token = reader.get_token()
    
    if token == 'blend':  # animation node is a blend-type node
        anim_node.param_name = reader.get_token()
        anim_node.min_value = read_float(reader)
        anim_node.max_value = read_float(reader)
        anim_node.blend_animations = []
        
        token = reader.get_token()
        
        if token == 'wrap':
            anim_node.flags = 1  # bitflags, but this is the only flag, so I'm gonna cheat a little bit
            token = reader.get_token()
        
        if token != '{':
            raise SparkException("Syntax error: expected { at line " + str(reader.get_line()) + " (got '" + token + "'")
        
        start_line = reader.get_line()
        while reader.peek_token() != '}':
            if not reader.has_token():
                raise SparkException("Syntax error: unexpected end of file while searching for a } "
                                     "to match the { at line " + str(start_line) + ".")
            blend_animation = read_animation_node(d, reader)
            """:type : int"""
            
            if blend_animation is None:
                return None
            
            anim_node.blend_animations.append(blend_animation)
        reader.get_token()  # read the } we were peeking before
    
    elif token == 'layer':
        anim_node.layer_animations = []
        if reader.get_token() != '{':
            raise SparkException("Syntax error: expected { at line " + str(reader.get_line()))
        start_line = reader.get_line()
        while reader.peek_token() != '}':
            if not reader.has_token():
                raise SparkException("Syntax error: unexpected end of file while searching for a } "
                                     "to match the { at line " + str(start_line) + ".")
            layer_animation = read_animation_node(d, reader)
            """:type : int"""
            
            if layer_animation is None:
                return None
            
            anim_node.layer_animations.append(layer_animation)
        reader.get_token()  # read the } we were peeking before
    
    else:
        # check to see if token refers to existing animation node
        animation_node_index = get_animation_node_by_name(d, token)
        if animation_node_index != -1:
            return animation_node_index
        
        # Assume it's the name of a scene
        animation = Animation()
        animation.source_name = token
        animation.flags = 0
        animation.start_frame = -1
        animation.end_frame = -1
        animation.speed = 1.0
        
        while True:
            token = reader.peek_token()
            
            if token == 'speed':
                reader.get_token()
                animation.speed = read_float(reader)
            
            elif token == 'loop':
                reader.get_token()
                animation.flags |= ANIMATION_FLAG_LOOPING()
            
            elif token == 'relative_to':
                raise SparkException("Sorry!  'relative_to' is not supported at this time.")
            
            elif token == 'relative_to_start':
                reader.get_token()
                animation.flags |= ANIMATION_FLAG_RELATIVE()
            
            elif token == 'from':
                reader.get_token()
                animation.start_frame = reader.get_token()  # can be integer frames, OR marker names
            
            elif token == 'to':
                reader.get_token()
                animation.end_frame = reader.get_token()  # can be integer frames, OR marker names
            
            else:
                break
        
        # See if an animation with the same properties already exists, to reuse
        index = find_equivalent_animation(d, animation)
        if index != -1:
            anim_node.animation = index
        else:
            anim_node.animation = len(d.animations)
            d.animations.append(animation)
    
    animation_node_index = len(d.animation_nodes)
    d.animation_nodes.append(anim_node)
    return animation_node_index


def parse_model_compile_list():
    multiple = bpy.data.texts.find('model_compile_list')
    if multiple == -1:
        return [], []  # empty list of model names, empty list of model_compile text block names

    model_list = bpy.data.texts[multiple].as_string()
    reader = TokenizedReader(model_list)

    model_name_list = []  # list of strings
    model_compile_list = []  # list of strings
    while reader.has_token():
        # read model name
        model_name = reader.get_token()
        if model_name == '':
            raise SparkException("Expected name of .model at line " + str(reader.get_line()) +
                                 ", got an empty string instead.")
        # clean up model name, ensuring that the name ends with ".model"
        model_name_split = model_name.split('.')
        if model_name_split[-1].lower() != 'model':
            model_name_split.append('model')
        model_name = '.'.join(model_name_split)
        if model_name in model_name_list:
            raise SparkException("Duplicate model name provided in model_compile_list block.  Aborting." +
                                 "(Error at line " + str(reader.get_line()) + ")")
        model_name_list.append(model_name)

        # read model_compile name
        if not reader.has_token():
            raise SparkException("Expected name of model_compile text block to compile model '" + model_name +
                                 "' at line " + str(reader.get_line()))
        model_compile_name = reader.get_token()
        if model_compile_name == '':
            raise SparkException("Expected name of model_compile text block at line " + str(reader.get_line()) +
                                 ", got an empty string instead.")
        if bpy.data.texts.find(model_compile_name) == -1:
            raise SparkException("Unable to locate text block named '" + model_compile_name +"'.")
        if model_compile_name in model_compile_list:
            raise SparkException("Duplicate model_compile name provided in model_compile_list block.  Aborting." +
                                 "(Error at line " + str(reader.get_line()) + ")")
        model_compile_list.append(model_compile_name)

    if not model_name_list:  # empty
        print(blend_name, ": A 'model_compile_list' block was provided, but empty!  Attempting single model export...")

    return model_name_list, model_compile_list


def parse_model_compile(d, model_compile_name):
    """
    :type model_datas: list[ModelData]
    """
    # Returns -1 if unsuccessful, 1 if successful

    available = bpy.data.texts.find(model_compile_name)
    if available == -1:
        return -1
    
    text = bpy.data.texts[available].as_string()
    reader = TokenizedReader(text)
    
    while reader.has_token():
        # read directives
        token = reader.get_token()

        if token == 'attach_point':
            name = reader.get_token()
            if name == '':
                raise SparkException("Expected name of attach_point at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            if bpy.data.objects.find(name) == -1:
                raise SparkException('Object "' + name + '" does''nt exist!  Check spelling. (Error at line ' +
                                     str(reader.get_line()) + ')')
            if name in d.attach_points:
                print(blend_name, ": Warning: duplicate attach_point declared at line ", reader.get_line(), ". Skipping.")
                continue
            else:
                d.attach_points.append(name)
            continue

        elif token == 'geometry':
            if d.geometry_group is not None:
                print(blend_name, ": Warning: duplicate geometry group declared at line ", reader.get_line(), ". Skipping.")
                continue
            name = reader.get_token()
            scene = reader.get_token()
            if name == '':
                raise SparkException("Expected name of geometry group at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            if bpy.data.groups.find(name) == -1:
                raise SparkException('Group "' + name + '" doesn''t exist!  Check spelling. (Error at line ' +
                                     str(reader.get_line()) + ')')
            if scene == '':
                raise SparkException("Expected name of geometry scene at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            if bpy.data.scenes.find(scene) == -1:
                raise SparkException('Scene "' + scene + '" doesn''t exist!  Check spelling. (Error at line ' +
                                     str(reader.get_line()) + ')')
            d.geometry_group = [name, scene]
            continue

        elif token == 'physics':
            if d.physics_groups is not None:
                print(blend_name, ": Warning: duplicate physics group declared at line ", reader.get_line(), ". Skipping.")
                continue
            # Two ways to do this: the old way -- declare a single physics group name, and the new way -- declare
            # several physics maps with rep names.  We know it's the new way if the first token is an open bracket.
            token = reader.get_token()
            if token == '{':  # new style
                while reader.peek_token() != '}':
                    if reader.peek_token() is None:
                        raise SparkException("Unexpected end of text block when reading physics data!  "
                                             "(Error at line " + str(reader.get_line()) + ")")
                    name = reader.get_token()
                    group = reader.get_token()
                    scene = reader.get_token()
                    if not group:
                        raise SparkException("Expected name of physics group at line " + str(reader.get_line()) +
                                             ", got empty string instead.")
                    if bpy.data.groups.find(group) == -1:
                        raise SparkException('Group "' + group + '" doesn''t exist!  Check spelling. (Error at line ' +
                                             str(reader.get_line()) + ')')
                    if not scene:
                        raise SparkException("Expected name of physics scene at line " + str(reader.get_line()) +
                                             ", got empty string instead.")
                    if bpy.data.scenes.find(scene) == -1:
                        raise SparkException('Scene "' + scene + '" doesn''t exist!  Check spelling. (Error at line ' +
                                             str(reader.get_line()) + ')')
                    if d.physics_groups is None:
                        d.physics_groups = []
                    d.physics_groups.append([name, group, scene])
                reader.get_token()  # skip the }
                continue
            else:  # old style physics
                group = token
                scene = reader.get_token()
                if not group:
                    raise SparkException("Expected name of physics group at line " + str(reader.get_line()) +
                                         ", got empty string instead.")
                if not scene:
                        raise SparkException("Expected name of physics scene at line " + str(reader.get_line()) +
                                             ", got empty string instead.")
                if bpy.data.groups.find(group) == -1:
                    raise SparkException('Group "' + group + '" doesn''t exist!  Check spelling. (Error at line ' +
                                         str(reader.get_line()) + ')')
                if bpy.data.scenes.find(scene) == -1:
                        raise SparkException('Scene "' + scene + '" doesn''t exist!  Check spelling. (Error at line ' +
                                             str(reader.get_line()) + ')')
                d.physics_groups = [["default", group, scene]]
                continue

        elif token == 'scale':
            if d.scale_value != 1.0:
                print(blend_name, ": Warning: 'scale' value declared multiple times.  Multiplying subsequent declarations.")
            d.scale_value *= read_float(reader)
            continue

        elif token == 'linear_max_error':
            lin_max = read_float(reader)
            if d.linear_max_error is not None:
                print(blend_name, ": Warning: duplicate linear_max_error value declared at line ", reader.get_line(), ". Skipping.")
                continue
            d.linear_max_error = lin_max
            continue

        elif token == 'quat_max_error':
            qt_max = read_float(reader)
            if d.quat_max_error is not None:
                print(blend_name, ": Warning: duplicate quat_max_error value declared at line ", reader.get_line(), ". Skipping.")
                continue
            d.quat_max_error = qt_max
            continue

        elif token == 'disable_compression':
            d.compression_enabled = False
            continue

        elif token == 'animation_model':
            if d.animation_model is not None:
                print(blend_name, ": Warning: duplicate animation_model declared at line ", reader.get_line(), ". Skipping.")
                continue
            anim_model = reader.get_token()
            if anim_model == '':
                raise SparkException("Expected path of animation_model at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            d.animation_model = anim_model
            continue

        elif token == 'collisions':
            pair = [False, None, None]
            if d.read_collision_pairs is None:
                d.read_collision_pairs = []
            c_token = reader.get_token()
            if c_token == 'on':
                pair[0] = True
            elif c_token == 'off':
                pair[0] = False
            else:
                raise SparkException('Expected "on" or "off" (w/o quotes) at line ' + str(reader.get_line()) + '.')
            
            solid1 = reader.get_token()
            if solid1 == '':
                raise SparkException("Expected name of collision solid #1 at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            if bpy.data.objects.find(solid1) == -1:
                raise SparkException('Object "' + solid1 + '" does''nt exist!  Check spelling. (Error at line ' +
                                     str(reader.get_line()) + ')')
            pair[1] = solid1
            
            solid2 = reader.get_token()
            if solid2 == '':
                raise SparkException("Expected name of collision solid #2 at line " + str(reader.get_line()) +
                                     ", got empty string instead.")
            if bpy.data.objects.find(solid2) == -1:
                raise SparkException('Object "' + solid2 + '" does''nt exist!  Check spelling. (Error at line ' +
                                     str(reader.get_line()) + ')')
            pair[2] = solid2
            
            d.read_collision_pairs.append(pair)
            continue
        
        elif token == "animation_node":
            name = reader.get_token()
            if d.animation_nodes.count(name) > 0:
                raise SparkException("Animation node " + name + " has already been defined. (Error at line " +
                                     str(reader.get_line()) + ".")
            animation_node = read_animation_node(d, reader)
            d.animation_nodes[animation_node].name = name
            continue
        
        elif token == "animation":
            sequence = Sequence()
            sequence.name = reader.get_token()
            sequence.animation_node = read_animation_node(d, reader)
            sequence.length = 0.0
            
            d.sequences.append(sequence)
            continue

        elif token == "alternate_origin":
            origin_name = reader.get_token()
            if bpy.data.objects.find(origin_name) == -1:
                raise SparkException("Alternate origin object '" + origin_name + "' doesn't exist!")
            d.alternate_origin_object = bpy.data.objects[origin_name]


        else:
            raise SparkException("Syntax Error: Unexpected token " + token + " at line " + str(reader.get_line()) + ".")
    
    # Check that required parameters were supplied or at least exist.
    if d.geometry_group is None:
        raise SparkException("Error: No geometry group was specified.  Aborting.")
    
    # insert some default values
    if d.linear_max_error is None:
        d.linear_max_error = 0.0001
    if d.quat_max_error is None:
        d.quat_max_error = 0.01

    d.scale_value /= INCHESPERMETER()

    # disable compression if it's a view model (ie the name of the blend file ends with '_view')
    f_name = '.'.join(bpy.data.filepath.replace('\\','/').split('/')[-1].split('.')[:-1])
    if len(f_name) >= 5 and f_name[-5:].lower() == '_view':
        d.compression_enabled = False
        print(blend_name, ": Disabling animation compression for view model.")

    return 1


def build_face_sets(d):
    """
    :type d: ModelData
    """
    face_sets = d.face_sets
    tris = d.model.triangles
    for i in range(len(tris)):
        material = tris[i].material
        # Find a face set with a matching material and not too many bones.
        if not face_sets:  # None or empty
            d.face_sets = []; face_sets = d.face_sets
            new_set = FaceSet()
            new_set.material_index = material
            face_sets.append(new_set)

        found_set = False
        face_bones = set()
        for k in range(3):
            vert_bone_list = tris[i].verts[k].get_bone_list(bone_offset=(1 if d.add_world_bone else 0))
            vert_bones = set(vert_bone_list)
            if not vert_bone_list and d.add_world_bone:  # if it's empty, we know this vert has no parent-bone(s)
                vert_bones.add(0)  # world-space bone index is always 0
            face_bones = face_bones | vert_bones
        for j in range(len(face_sets)):
            if face_sets[j].material_index == tris[i].material:
                # need to figure out if this face can be added without going over the bone limit
                set_bones = face_sets[j].bones
                new_bones = list(face_bones.union(set_bones))
                if len(new_bones) < MAX_BONES_PER_FACE_SET():
                    # not sure why this is < instead of <=, but that's the way it is in the spark source code, so I'm
                    # just going to go with that.
                    found_set = True
                    face_sets[j].faces.append(i)
                    face_sets[j].bones = new_bones
                    break

        if not found_set:
            # No suitable set was found, add one
            new_set = FaceSet()
            new_set.material_index = tris[i].material
            new_set.faces.append(i)
            new_set.bones = list(face_bones)
            face_sets.append(new_set)


def estimate_size(d: ModelData):
    # returns the final size of the model, in bytes
    size = 0

    size += 4  # MDL and version number

    # vertex chunk
    size += 8  # chunk header
    size += 4  # number of vertices
    size += 92 * len(d.model.verts)  # 92 bytes per vertex

    # indices chunk
    size += 8  # chunk header
    size += 4  # number of indices
    size += 4 * len(d.model.triangles) * 3  # 12 bytes per triangle ( 4 per indice, 3 indices per triangle )

    # face-sets chunk
    size += 8  # chunk header
    size += 4  # number of face sets
    for i in range(len(d.face_sets)):
        size += 16  # 4 bytes each: material index, first face index, num faces, num bones
        size += 4 * len(d.face_sets[i].bones)  # 4 bytes for each bone

    # bones chunk
    size += 8  # chunk header
    size += 4  # number of bones
    for i in range(len(d.model.bones)):
        size += 4 + len(d.model.bones[i].name)  # name string ( 4 bytes for length, + 1 bytes per character )
        size += 4  # parent index
        size += 60  # affine parts for bone
    for i in range(len(d.model.extra_bones)):
        size += 4 + len(d.model.extra_bones[i].name)
        size += 4
        size += 60
    if d.add_world_bone:
        size += 4 + len('world-space')
        size += 4
        size += 60

    # materials chunk
    size += 8  # chunk header
    size += 4  # number of materials
    for i in range(len(d.model.materials)):
        size += 4 + len(d.model.materials[i].spark_material)  # path string

    # animations chunk
    if d.animations:
        size += 8  # chunk header
        size += 4  # number of animations
        for i in range(len(d.animations)):
            size += 16  # 4 ea: flags, frame count, framerate, compression boolean
            if d.animations[i].compressed_animation:
                size += 4  # pose curves count
                for j in range(len(d.animations[i].compressed_animation.pose_curves)):
                    size += 20  # number of keys in each curve (4 * 5)
                    # 4 floats (4 bytes each) per position curve point
                    size += 16 * len(d.animations[i].compressed_animation.pose_curves[j].position_curve.c_keys_x)
                    # 4 floats (4 bytes each) per scale curve point
                    size += 16 * len(d.animations[i].compressed_animation.pose_curves[j].scale_curve.c_keys_x)
                    # 2 floats (4 bytes each) per flip curve point
                    size += 8 * len(d.animations[i].compressed_animation.pose_curves[j].flip_curve.c_keys_x)
                    # 5 floats (4 bytes each) per rotation curve point
                    size += 20 * len(d.animations[i].compressed_animation.pose_curves[j].rotation_curve.c_keys_x)
                    # 5 floats (4 bytes each) per scale-rotation curve point
                    size += 20 * len(d.animations[i].compressed_animation.pose_curves[j].scale_rotation_curve.c_keys_x)
            # full keyframe data
            size += 4  # number of bone animations
            for j in range(len(d.animations[i].bone_animations)):
                size += 4  # bone index
                size += 60 * len(d.animations[i].bone_animations[j].keys)

            size += 4  # number of frame tags
            for j in range(len(d.animations[i].frame_tags)):
                size += 4  # frame
                size += 4 + len(d.animations[i].frame_tags[j].name)  # name string

    # animation nodes chunk
    if d.animation_nodes:
        size += 8  # chunk header
        size += 4  # number of animation nodes

        for i in range(len(d.animation_nodes)):
            if d.animation_nodes[i].animation != -1:
                size += 4  # node type
                size += 4  # flags
                size += 4  # animation index
            elif d.animation_nodes[i].blend_animations:
                size += 4  # node type
                size += 4  # flags
                size += 4  # blend parameters index
                size += 8  # min and max values
                size += 4  # number of blended animations
                size += 4 * len(d.animation_nodes[i].blend_animations)
            else:
                size += 4  # node type
                size += 4  # flags
                size += 4  # number of animation layers
                size += 4 * len(d.animation_nodes[i].layer_animations)

    # sequences chunk
    if d.sequences:
        size += 8  # chunk header
        size += 4  # number of sequences

        for i in range(len(d.sequences)):
            size += 4 + len(d.sequences[i].name)  # name string
            size += 4  # animation node index
            size += 4  # sequence length

    # blend parameters chunk
    if d.blend_parameters:
        size += 8  # chunk header
        size += 4  # number of blend parameters

        for i in range(len(d.blend_parameters)):
            size += 4 + len(d.blend_parameters[i])

    # collision rep entries and collision rep chunks
    if d.collision_rep_entries:
        size += 8  # chunk header (collision reps)
        size += 4  # number of collision reps
        size += 24 * len(d.collision_reps)  # 24 bytes per collision rep

        size += 4  # number of collision rep entries
        for i in range(len(d.collision_rep_entries)):
            size += 4 + len(d.collision_rep_entries[i].name)  # name string
            size += 4  # rep index

    # solids chunk
    if d.solids:
        size += 8  # chunk header
        size += 4  # number of solids
        for i in range(len(d.solids)):
            size += 4 + len(d.solids[i].name)  # name string
            size += 4  # bone index
            size += 48  # object to bone coords
            size += 4  # mass
            size += 4  # number of vertices
            size += 12 * len(d.solids[i].vertices)  # 12 bytes per solid-vertex
            size += 4  # number of triangles
            size += 12 * len(d.solids[i].triangles)  # 12 bytes per solid-triangle

    # joints chunk
    if d.joints:
        size += 8  # chunk header
        size += 4  # number of joints
        for i in range(len(d.joints)):
            size += 4 + len(d.joints[i].name)  # name string
            size += 8  # solid 1/2 index
            size += 96  # solid 1/2 coords
            size += 4  # dummy field for joint-type (future expansion)
            size += 24  # angle limits

    # collision pairs chunk
    write_pairs = False
    for i in range(len(d.collision_pairs)):
        if not d.collision_pairs[i].enabled:
            write_pairs = True
            break
    if d.collision_pairs and write_pairs:
        size += 8  # chunk header
        size += 4  # num pairs
        for i in range(len(d.collision_pairs)):
            if not d.collision_pairs[i].enabled:
                size += 8  # collision pair solid indices

    # attach points chunk
    if d.attach_point_objects:
        size += 8  # chunk header
        size += 4  # num points
        for i in range(len(d.attach_point_objects)):
            size += 4 + len(d.attach_point_objects[i].name)  # name string
            size += 4  # bone index
            size += 48  # coords

    # cameras chunk
    if d.cameras:
        size += 8  # chunk header
        size += 4  # num cameras
        for i in range(len(d.cameras)):
            size += 4 + len(d.cameras[i].name)  # name string
            size += 4  # bone index
            size += 4  # horizontal fov
            size += 48  # coords

    # bounding box chunk
    size += 8  # chunk header
    size += 24  # min max vec3s

    # bone bounding box chunks
    if d.model.bone_bounds:
        size += 8  # chunk header
        # 12 bytes per vec3, 2 vec3s per bound box, 1 bound box per bone
        size += 24 * (len(d.model.bones) + (1 if d.add_world_bone else 0) + len(d.model.extra_bones))

    # animation model chunk
    if d.animation_model:
        size += 8  # chunk header
        size += 4 + len(d.animation_model)

    return size


def list_blend_parameters(d: ModelData):
    blend_params = set()
    for a in d.animation_nodes:
        if a.blend_animations:
            blend_params.add(a.param_name)
    d.blend_parameters = list(blend_params)


#def write_model(d: ModelData, base_dir: str, model_name: str):
def write_model(d: ModelData, model_name: str):
    build_face_sets(d)
    list_blend_parameters(d)

    est_size = estimate_size(d)
    writer = SparkWriter()
    writer.allocate(est_size)

    writer.write_raw(b'MDL\x07')  # MDL version 7

    bone_offset = 1 if d.add_world_bone else 0  # extra bone for a static, 'world-coords' bone

    # vertex chunk
    writer.begin_chunk("Chunk_Vertices")
    writer.write_int32(len(d.model.verts))
    for i in range(len(d.model.verts)):
        d.model.verts[i].written_index = i
        writer.write_vertex(d.model.verts[i], bone_offset=bone_offset)
    writer.end_chunk()

    # indices chunk
    writer.begin_chunk("Chunk_Indices")
    writer.write_int32(len(d.model.triangles) * 3)
    for i in range(len(d.face_sets)):
        for j in range(len(d.face_sets[i].faces)):
            tri = d.model.triangles[d.face_sets[i].faces[j]]
            writer.write_int32(tri.verts[0].written_index)
            writer.write_int32(tri.verts[1].written_index)
            writer.write_int32(tri.verts[2].written_index)
    writer.end_chunk()

    # face sets chunk
    writer.begin_chunk("Chunk_FaceSets")
    writer.write_int32(len(d.face_sets))
    first_face = 0
    for i in range(len(d.face_sets)):
        writer.write_int32(d.face_sets[i].material_index)
        writer.write_int32(first_face)
        writer.write_int32(len(d.face_sets[i].faces))
        writer.write_int32(len(d.face_sets[i].bones))
        first_face += len(d.face_sets[i].faces)
        for j in range(len(d.face_sets[i].bones)):
            writer.write_int32(d.face_sets[i].bones[j])
    writer.end_chunk()

    # bones chunk
    writer.begin_chunk("Chunk_Bones")
    writer.write_int32(len(d.model.bones) + len(d.model.extra_bones) + bone_offset)
    if bone_offset:
        writer.write_string('world-space')
        writer.write_int32(0xFFFFFFFF)
        static_parts = AffineParts()
        static_parts.translation = Vec3(); static_parts.translation.data = [0.0, 0.0, 0.0]
        static_parts.rotation = Quat(); static_parts.rotation.data = [0.0, 0.0, 0.0, 1.0]
        static_parts.scale = Vec3(); static_parts.scale.data = [1.0, 1.0, 1.0]
        static_parts.scale_rotation = Quat(); static_parts.scale_rotation.data = [0.0, 0.0, 0.0, 1.0]
        static_parts.flip = 1.0
        writer.write_affine_parts(static_parts)
    for i in range(len(d.model.bones)):  # write blender bones
        writer.write_string(d.model.bones[i].name)
        if d.model.bones[i].parent:
            writer.write_int32(d.model.bone_to_index[d.model.bones[i].parent.name] + bone_offset)  # parent index
        else:
            writer.write_int32(0xFFFFFFFF)  # no parent
        writer.write_affine_parts(decompose_affine(d.model.bone_base_poses[i]))
    for i in range(len(d.model.extra_bones)):
        writer.write_string(d.model.extra_bones[i].name)
        writer.write_int32(0xFFFFFFFF)  # extra bones never have parents
        writer.write_affine_parts(decompose_affine(d.model.extra_bones[i].bone_to_world_matrix))
    writer.end_chunk()

    # materials chunk
    writer.begin_chunk("Chunk_Materials")
    writer.write_int32(len(d.model.materials))
    for i in range(len(d.model.materials)):
        writer.write_string(d.model.materials[i].spark_material)
    writer.end_chunk()

    # animations chunk
    if d.animations:
        writer.begin_chunk("Chunk_Animations")
        writer.write_int32(len(d.animations))
        for i in range(len(d.animations)):
            writer.write_int32(d.animations[i].flags)
            writer.write_int32(len(d.animations[i].bone_animations[0]))
            writer.write_float(d.animations[i].frame_rate * d.animations[i].speed)
            writer.write_bool(d.animations[i].compressed_animation is not None)

            # write compressed animation
            if d.animations[i].compressed_animation:
                c_anim = d.animations[i].compressed_animation
                writer.write_int32(len(c_anim.pose_curves))
                for j in range(len(c_anim.pose_curves)):
                    # pos curve
                    pos_curve = c_anim.pose_curves[j].position_curve
                    writer.write_int32(len(pos_curve.c_keys_x))
                    for k in range(len(pos_curve.c_keys_x)):
                        writer.write_float(pos_curve.c_keys_x[k])
                    for k in range(len(pos_curve.c_keys_x)):
                        writer.write_vec3(pos_curve.c_keys_y[k])
                    del pos_curve

                    # scale curve
                    scale_curve = c_anim.pose_curves[j].scale_curve
                    writer.write_int32(len(scale_curve.c_keys_x))
                    for k in range(len(scale_curve.c_keys_x)):
                        writer.write_float(scale_curve.c_keys_x[k])
                    for k in range(len(scale_curve.c_keys_x)):
                        writer.write_vec3(scale_curve.c_keys_y[k])
                    del scale_curve

                    # flip curve
                    flip_curve = c_anim.pose_curves[j].flip_curve
                    writer.write_int32(len(flip_curve.c_keys_x))
                    for k in range(len(flip_curve.c_keys_x)):
                        writer.write_float(flip_curve.c_keys_x[k])
                    for k in range(len(flip_curve.c_keys_x)):
                        writer.write_float(flip_curve.c_keys_y[k])
                    del flip_curve

                    # rotation curve
                    rot_curve = c_anim.pose_curves[j].rotation_curve
                    writer.write_int32(len(rot_curve.c_keys_x))
                    for k in range(len(rot_curve.c_keys_x)):
                        writer.write_float(rot_curve.c_keys_x[k])
                    for k in range(len(rot_curve.c_keys_x)):
                        writer.write_quat(rot_curve.c_keys_y[k])
                    del rot_curve

                    # scale-rotation curve
                    s_rot_curve = c_anim.pose_curves[j].scale_rotation_curve
                    writer.write_int32(len(s_rot_curve.c_keys_x))
                    for k in range(len(s_rot_curve.c_keys_x)):
                        writer.write_float(s_rot_curve.c_keys_x[k])
                    for k in range(len(s_rot_curve.c_keys_x)):
                        writer.write_quat(s_rot_curve.c_keys_y[k])
                    del s_rot_curve

            # full keyframe data
            writer.write_int32(len(d.animations[i].bone_animations))
            for j in range(len(d.animations[i].bone_animations)):
                bone_anim = d.animations[i].bone_animations[j]
                writer.write_int32(j + bone_offset)  # bone index - bone offset == bone animation index
                for k in range(len(d.animations[i].bone_animations[0])):
                    writer.write_affine_parts(bone_anim.keys[k])

            # write frame tags
            frame_tags = d.animations[i].frame_tags
            writer.write_int32(len(frame_tags))
            for j in range(len(frame_tags)):
                writer.write_int32(frame_tags[j].frame)
                writer.write_string(frame_tags[j].name)
        writer.end_chunk()

    # animation nodes chunk
    if d.animation_nodes:
        for i in range(len(d.animation_nodes)):
            d.animation_nodes[i].index = i
        param_name_to_index = {}
        for i in range(len(d.blend_parameters)):
            param_name_to_index[d.blend_parameters[i]] = i
        writer.begin_chunk("Chunk_AnimationNodes")
        writer.write_int32(len(d.animation_nodes))

        for i in range(len(d.animation_nodes)):
            node = d.animation_nodes[i]
            if node.animation != -1:  # regular animation
                writer.write_int32(ANIMATION_NODE_TYPE_ANIMATION())
                writer.write_int32(node.flags)
                writer.write_int32(node.animation)
            elif node.blend_animations:  # blend animation
                writer.write_int32(ANIMATION_NODE_TYPE_BLEND())
                writer.write_int32(node.flags)
                writer.write_int32(param_name_to_index[node.param_name])
                writer.write_float(node.min_value)
                writer.write_float(node.max_value)
                writer.write_int32(len(node.blend_animations))
                for j in range(len(node.blend_animations)):
                    writer.write_int32(node.blend_animations[j])
            else:  # layer animation
                writer.write_int32(ANIMATION_NODE_TYPE_LAYER())
                writer.write_int32(node.flags)
                writer.write_int32(len(node.layer_animations))
                for j in range(len(node.layer_animations)):
                    writer.write_int32(node.layer_animations[j])
        writer.end_chunk()

    # sequences chunk
    if d.sequences:
        writer.begin_chunk("Chunk_Sequences")
        writer.write_int32(len(d.sequences))
        for i in range(len(d.sequences)):
            seq = d.sequences[i]
            writer.write_string(seq.name)
            writer.write_int32(seq.animation_node)
            writer.write_float(seq.length)
        writer.end_chunk()

    # blend parameters chunk
    if d.blend_parameters:
        writer.begin_chunk("Chunk_BlendParameters")
        writer.write_int32(len(d.blend_parameters))
        for i in range(len(d.blend_parameters)):
            writer.write_string(d.blend_parameters[i])
        writer.end_chunk()

    # collision reps chunk and collision rep entries chunk
    if d.collision_rep_entries:
        writer.begin_chunk("Chunk_CollisionReps")
        writer.write_int32(len(d.collision_reps))
        for i in range(len(d.collision_reps)):
            rep = d.collision_reps[i]
            writer.write_int32(rep.num_solids)
            writer.write_int32(rep.first_solid_index)
            writer.write_int32(rep.num_joints)
            writer.write_int32(rep.first_joint_index)
            writer.write_int32(rep.num_pairs)
            writer.write_int32(rep.first_pair_index)

        writer.write_int32(len(d.collision_rep_entries))
        for i in range(len(d.collision_rep_entries)):
            writer.write_string(d.collision_rep_entries[i].name)
            writer.write_int32(d.collision_rep_entries[i].collision_rep_index)
        writer.end_chunk()

    # solids chunk
    if d.solids:
        solid_to_index = {}
        for i in range(len(d.solids)):
            solid_to_index[d.solids[i]] = i
        writer.begin_chunk("Chunk_Solids")
        writer.write_int32(len(d.solids))
        for i in range(len(d.solids)):
            sol = d.solids[i]
            writer.write_string(sol.name)
            if sol.bone_name:
                writer.write_int32(d.model.bone_to_index[sol.bone_name] + bone_offset)
            else:
                writer.write_int32(0xFFFFFFFF)  # no parent bone
            writer.write_coords(sol.object_to_bone_coords)
            writer.write_float(sol.mass)
            writer.write_int32(len(sol.vertices))
            for j in range(len(sol.vertices)):
                writer.write_vec3(sol.vertices[j])
            writer.write_int32(len(sol.triangles))
            for j in range(len(sol.triangles)):
                writer.write_int32(sol.triangles[j][0])
                writer.write_int32(sol.triangles[j][1])
                writer.write_int32(sol.triangles[j][2])
        writer.end_chunk()

    # joints chunk
    if d.joints:
        writer.begin_chunk("Chunk_Joints")
        writer.write_int32(len(d.joints))
        for i in range(len(d.joints)):
            j = d.joints[i]
            writer.write_string(j.name)

            writer.write_int32(solid_to_index[j.solid_1])
            writer.write_coords(j.joint_to_solid_1_coords)
            writer.write_int32(solid_to_index[j.solid_2])
            writer.write_coords(j.joint_to_solid_2_coords)

            writer.write_int32(0)  # leaving room for future expansion, could specify joint-type here.

            writer.write_vec3(j.minimum_angles)
            writer.write_vec3(j.maximum_angles)
        writer.end_chunk()

    # collision pairs chunk
    if d.collision_pairs:
        disabled_pairs = [p for p in d.collision_pairs if not p.enabled]
        if disabled_pairs:
            writer.begin_chunk("Chunk_CollisionPairs")
            writer.write_int32(len(disabled_pairs))
            for i in range(len(disabled_pairs)):
                writer.write_int32(solid_to_index[disabled_pairs[i].solid_1])
                writer.write_int32(solid_to_index[disabled_pairs[i].solid_2])
            writer.end_chunk()

    # attach points chunk
    if d.attach_point_objects:
        writer.begin_chunk("Chunk_AttachPoints")
        writer.write_int32(len(d.attach_point_objects))
        for i in range(len(d.attach_point_objects)):
            pt = d.attach_point_objects[i]
            writer.write_string(pt.name)
            writer.write_int32(d.model.bone_to_index[pt.parent_bone] + bone_offset if pt.parent_bone else 0xFFFFFFFF)
            writer.write_coords(pt.coords)
        writer.end_chunk()

    # cameras chunk
    if d.cameras:
        writer.begin_chunk("Chunk_Cameras")
        writer.write_int32(len(d.cameras))
        for i in range(len(d.cameras)):
            cam = d.cameras[i]
            writer.write_string(cam.name)
            if cam.parent_bone_blender:  # parented to pre-existing bone, not extra bone created exclusively for camera
                writer.write_int32(d.model.bone_to_index[cam.parent_bone_blender] + bone_offset)
            else:  # parented to a bone created just for this camera
                writer.write_int32(len(d.model.bones) + cam.parent_bone_extra + bone_offset)
            writer.write_float(cam.fov)
            writer.write_coords(cam.coords)
        writer.end_chunk()

    # bounding box chunk
    writer.begin_chunk("Chunk_BoundingBox")
    bounds = BoundBox(d.model.bound_box)
    writer.write_vec3(bounds.origin)
    writer.write_vec3(bounds.extents)
    writer.end_chunk()

    # bone bounding boxes
    if d.model.bone_bounds:
        writer.begin_chunk("Chunk_BoneBoundingBoxes")
        if bone_offset:
            writer.write_vec3(Vec3([0.0, 0.0, 0.0]))
            writer.write_vec3(Vec3([0.0, 0.0, 0.0]))
        for i in range(len(d.model.bone_bounds)):
            writer.write_vec3(d.model.bone_bounds[i].origin)
            writer.write_vec3(d.model.bone_bounds[i].extents)
        for i in range(len(d.model.extra_bones)):
            writer.write_vec3(Vec3([0.0, 0.0, 0.0]))
            writer.write_vec3(Vec3([0.0, 0.0, 0.0]))
        writer.end_chunk()

    # animation model chunk
    if d.animation_model:
        writer.begin_chunk("Chunk_AnimationModel")
        writer.write_string(d.animation_model)
        writer.end_chunk()
    
    thisDir = bpy.data.filepath.replace('\\','/').split('/')
    thisDir.pop() # discard filename
    thisDir.extend(model_name.split('/')) # add model filenames to end of base directory
    
    # loop through folders, changing last instance of keywords to the correct term for
    # the output directory.
    for i in range(len(thisDir)-1,-1,-1):
        if thisDir[i] == 'source':
            thisDir[i] = 'output'
            break;
    
    for i in range(len(thisDir)-1,-1,-1):
        if thisDir[i] == 'modelsrc':
            thisDir[i] = 'models'
            break;
    
    #ensure filename is .model
    file = thisDir.pop().split('.')
    if len(file) == 1:
        file.append('model')
    
    else:
        file.pop()
        file.append('model')
    
    thisDir.append('.'.join(file))
    
    out_file = '/'.join(thisDir)
    
    with open(out_file, 'wb') as file_write:
        file_write.write(writer.close_and_return())


#def save(base_dir: str):
def save():
    # old way
    # d = ModelData()
    # compile_success = parse_model_compile(d)

    # new way, allows for multiple .models output from single .blend, useful for prop variants (eg catwalks)
    # will automatically run the old parse_model_compile() if only one model is to be exported
    model_name_list, model_compile_list = parse_model_compile_list()
    # if compile_success == -1:
        # raise SparkException("No model_compile text-block found.  Aborting.")
    
    if not model_name_list:  # Empty
        filepath = bpy.data.filepath.replace('\\','/')
        name_split = filepath.split('/')
        name = name_split[-1].split('.')
        if name[-1] == 'blend':
            name.pop()
            name.append('model')
        name = '.'.join(name)
        model_name_list.append(name)
        model_compile_list.append('model_compile')
    
    success_list = [True] * len(model_name_list)
    for i in range(len(model_name_list)):
        try:
            d = ModelData()
            compile_success = parse_model_compile(d, model_compile_list[i])
            if compile_success == -1:
                print(blend_name, ": Unable to locate the model_compile text-block '",
                      model_compile_list[i], "'.  This model failed, but will attempt to compile others",
                      sep='', file=sys.stderr)
                success_list[i] = False
                continue
            load_geometry(d)
            load_animations(d)
            load_physics(d)
            write_model(d, model_name_list[i])
        except SparkException as e:
            success_list[i] = False
            print(blend_name, ": SparkException raised!  ", e.args[0], sep='', file=sys.stderr)
        
        except Exception as e2:
            success_list[i] = False
            print(blend_name, ": Exception raised!  ", e2.__class__.__name__, ": ", e2.args[0], sep='', file=sys.stderr)
           
    succeeded_count = 0
    failed_count = 0
    for i in range(len(model_name_list)):
        if success_list[i] == True:
            succeeded_count += 1
        else:
            failed_count += 1
    
    if succeeded_count == 0 and failed_count == 0:
        print(blend_name, ": No models were found/built!", file=sys.stderr)
    elif succeeded_count > 0 and failed_count == 0:
        print(blend_name, ": All ", succeeded_count, " files successfully built.", sep='', file=sys.stderr)
    elif succeeded_count == 0 and failed_count > 0:
        print(blend_name, ": All ", failed_count, " files failed to build.", sep='', file=sys.stderr)
    else:
        print(blend_name, ": ", succeeded_count, " files built successfully, but ", failed_count, " files failed to build.", sep='', file=sys.stderr)
    
    if failed_count > 0:
        print(blend_name, ": Failed files:", sep='', file=sys.stderr)
        for i in range(len(model_name_list)):
            if success_list[i] == False:
                print("  ", model_name_list[i], sep='', file=sys.stderr)
        
    sys.exit(1)
    
    
    
    
    
    
    
    

