# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

# Physics-related stuff goes in here

import bpy
import mathutils
from spark_common import *
from export_spark_model import *


# noinspection PyPep8Naming
def SUPPORTED_JOINT_TYPES(): return ['POINT', 'FIXED', 'HINGE', 'GENERIC']


class CollisionRepEntry:
    def __init__(self):
        self.name = ''
        self.collision_rep_index = -1


class CollisionRep:
    def __init__(self):
        self.num_solids = 0
        self.first_solid_index = 0
        self.num_joints = 0
        self.first_joint_index = 0
        self.num_pairs = 0
        self.first_pair_index = 0


class Plane:
    def __init__(self, vert: Vec3, normal: Vec3):
        self.a = normal.x
        self.b = normal.y
        self.c = normal.z
        self.d = -vert.dot_product(normal)

    def __getattr__(self, item):
        if item == 'normal':
            return Vec3(self.a, self.b, self.c)
        else:
            return None


class CollisionMesh:
    """
    Solid given in world space.  Only used to check two solids for overlap.
    """
    def __init__(self, solid, transform=None):
        self.vertices = []
        """:type : list(Vec3)"""
        self.planes = []
        """:type : list(Plane)"""

        self.solid_link = solid  # Link back to original solid, for convenience
        """:type : Solid"""

        self.min_extents = None
        """:type : Vec3"""
        self.max_extents = None
        """:type : Vec3"""

        if transform:
            self.vertices = [transform * solid.vertices[i] for i in range(len(solid.vertices))]
        else:
            self.vertices = [v for v in solid.vertices]

        # noinspection PyUnusedLocal
        self.planes = [None for i in range(len(solid.triangles))]
        num_planes = 0
        for t in solid.triangles:
            v0 = solid.vertices[t[0]]
            v1 = solid.vertices[t[1]]
            v2 = solid.vertices[t[2]]

            normal, mag = cross_product(v1 - v0, v2 - v0).normalized_and_mag()
            if mag > 0.0:
                self.planes[num_planes] = Plane(v0, normal)
                num_planes += 1
        self.planes = self.planes[:num_planes]  # we over-allocated assuming every triangle would yield a valid plane.

    def process_bound_box(self):
        self.min_extents = Vec3()
        self.max_extents = Vec3()

        self.min_extents.x = min(self.vertices, key=lambda v: v.x).x
        self.min_extents.y = min(self.vertices, key=lambda v: v.y).y
        self.min_extents.z = min(self.vertices, key=lambda v: v.z).z
        self.max_extents.x = max(self.vertices, key=lambda v: v.x).x
        self.max_extents.y = max(self.vertices, key=lambda v: v.y).y
        self.max_extents.z = max(self.vertices, key=lambda v: v.z).z


class CollisionPair:
    def __init__(self):
        self.enabled = False
        """:type : bool"""
        self.solid_1 = None
        """:type : Solid"""
        self.solid_2 = None
        """:type : Solid"""


class Solid:
    def __init__(self):
        self.bone_name = None
        """:type : str"""
        self.name = ''
        """:type : str"""
        self.parent_name = ''  # name of parent bone
        """:type : str"""
        self.object_to_bone_coords = None
        """:type : Coords"""
        self.mass = 1.0
        """:type : float"""

        self.vertices = []
        """:type : list[Vec3]"""
        self.triangles = []
        """:type : list[list[int]]"""

        self.index = -1


class Joint:
    def __init__(self):
        self.name = ''
        """:type : str"""
        self.minimum_angles = None
        """:type : Vec3"""
        self.maximum_angles = None
        """:type : Vec3"""
        self.solid_1 = None
        """:type : Solid"""
        self.solid_2 = None
        """:type : Solid"""
        self.joint_to_solid_1_coords = None
        """:type : Coords"""
        self.joint_to_solid_2_coords = None
        """:type : Coords"""

        self.index = -1


def get_solid_from_name(d, name):
    """
    @type d: ModelData
    @type name: str
    """
    for i in range(0, len(d.solids)):
        if d.solids[i].name == name:
            return d.solids[i]
    return None


def add_solid(d, scene, obj):
    new_solid = Solid()
    transform = None

    temp_obj = bpy.data.objects.new('temp_solid_processing_object',
                                    bpy.data.meshes.new_from_object(scene, obj, True, 'PREVIEW'))
    temp_mesh = temp_obj.data
    scene.objects.link(temp_obj)
    scene.objects.active = temp_obj
    bpy.context.screen.scene = scene

    # noinspection PyCallByClass
    bpy.ops.object.mode_set(mode='EDIT', toggle=False)
    bpy.ops.mesh.reveal()
    # noinspection PyCallByClass,PyTypeChecker
    bpy.ops.mesh.select_all(action='SELECT')
    # noinspection PyCallByClass,PyTypeChecker
    bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=False,
                             make_holes=False, join_triangles=False)
    bpy.ops.mesh.faces_shade_flat()
    bpy.ops.mesh.normals_make_consistent()
    # noinspection PyCallByClass
    bpy.ops.object.mode_set(mode='OBJECT', toggle=False)

    if -0.000001 < obj.scale.x < 0.000001:  # zero scale is not allowed
        return None
    if -0.000001 < obj.scale.y < 0.000001:  # zero scale is not allowed
        return None
    if -0.000001 < obj.scale.z < 0.000001:  # zero scale is not allowed
        return None

    scale_mat = mathutils.Matrix()
    # Cancel out scale so it can be applied to the mesh directly
    scale_mat[0][0] = obj.scale.x
    scale_mat[1][1] = obj.scale.y
    scale_mat[2][2] = obj.scale.z

    # scale up vertices to compensate for resetting the object's scale to 1.0
    temp_mesh.transform(scale_mat * d.scale_value)

    reverse_winding = False
    if (obj.scale.x * obj.scale.y * obj.scale.z) < 0.0:
        reverse_winding = True

    valid_parent = False
    # Calculate the object-to-bone transform
    if obj.parent:
        if obj.parent_type != 'BONE':
            # the user may have been attempting to parent to a bone in the armature.  Warn them of this simple mistake.
            if obj.parent.type == 'ARMATURE':
                print("Warning: Solid '", obj.name, "' is parented to the armature OBJECT, not a particular "
                                                    "bone.  Export continuing under the assumption that this solid is "
                                                    "static, and un-bound to any bone.", sep='')
                # get the world space transform of the object.  It's static unless it's parented to a bone.
                transform = Mat4(obj.matrix_world)
                # transform to alternate origin, if applicable
                if d.alternate_origin_object:
                    transform = Mat4(d.alternate_origin_object.matrix_world.inverted()) * transform

        else:  # Parented to bone
            if d.alternate_origin_object:
                solid_transform = Mat4(d.alternate_origin_object.matrix_world.inverted()) * Mat4(obj.matrix_world)
            else:
                solid_transform = Mat4(obj.matrix_world)
            transform = d.model.bone_world_mats[d.model.bone_to_index[obj.parent_bone]].get_inverse()\
                        * solid_transform

            new_solid.bone_name = obj.parent_bone
            valid_parent = True
    else:
        transform = Mat4(obj.matrix_world)
        # transform to alternate origin, if applicable
        if d.alternate_origin_object:
            transform = Mat4(d.alternate_origin_object.matrix_world.inverted()) * transform

    transform *= Mat4(scale_mat.inverted())

    if valid_parent:
        transform_mat = transform
    else:
        transform_mat = transform
        transform_mat.fix_axes(reverse=True)
    new_solid.object_to_bone_coords = Coords(transform_mat)
    new_solid.object_to_bone_coords.origin *= d.scale_value

    new_solid.vertices = [Vec3([v.co[0], v.co[1], v.co[2]]) for v in temp_mesh.vertices]

    if reverse_winding:  # Mesh is inside-out (odd-number of negative scale components)
        new_solid.triangles = [[p.vertices[0], p.vertices[2], p.vertices[1]] for p in temp_mesh.polygons]
    else:
        new_solid.triangles = [[p.vertices[0], p.vertices[1], p.vertices[2]] for p in temp_mesh.polygons]

    scene.objects.unlink(temp_obj)
    bpy.data.objects.remove(temp_obj)
    bpy.data.meshes.remove(temp_mesh)

    new_solid.name = obj.name

    if obj.rigid_body:  # if the object has rigid body settings, get the mass, otherwise leave it at the default 1.0
        new_solid.mass = obj.rigid_body.mass

    return new_solid


def add_joint(d, obj, solid_names):
    assert obj.type == 'EMPTY'

    if not obj.rigid_body_constraint:
        raise SparkException("Joint '" + obj.name + "' has no rigid_body_constraint settings!  "
                             "Ensure you've setup the constraint in the object's 'physics' tab.")

    constraint = obj.rigid_body_constraint

    if constraint.type not in SUPPORTED_JOINT_TYPES():
        raise SparkException("Rigid body constraint '" + constraint.type + "' is not a supported type!  "
                             "Please change the constraint type for Joint '" + obj.name + "' to a supported value.  "
                             "Supported values are: " + ', '.join(SUPPORTED_JOINT_TYPES()[:-1]) + ', and '
                             + SUPPORTED_JOINT_TYPES()[-1] + '.')

    # Skip unless both objects supplied
    if not constraint.object1:
        print("Warning: Skipping joint '", obj.name, "'.  Solid 1 was not defined.")
        return None
    if not constraint.object2:
        print("Warning: Skipping joint '", obj.name, "'.  Solid 2 was not defined.")
        return None

    # Skip unless both objects are valid solids
    if constraint.object1.name not in solid_names:
        print("Warning: Skipping joint '", obj.name, "'.  Solid 1 was a valid solid defined in the collision rep.")
        return None
    if constraint.object2.name not in solid_names:
        print("Warning: Skipping joint '", obj.name, "'.  Solid 2 was a valid solid defined in the collision rep.")
        return None

    # Skip disabled constraints
    if not constraint.enabled:
        return None

    # Add solid1 and 2 as collision pairs if "Disable Collisions" is enabled for this joint
    if constraint.disable_collisions:
        new_pair = CollisionPair()
        new_pair.enabled = False
        new_pair.solid_1 = get_solid_from_name(d, constraint.object1.name)
        new_pair.solid_2 = get_solid_from_name(d, constraint.object2.name)
        # Search to see if this collision pair is already defined.  If so, skip it.
        found = False
        for i in range(len(d.collision_pairs)):
            pair = d.collision_pairs[i]
            if pair.solid_1 == new_pair.solid_1 and pair.solid_2 == new_pair.solid_2:
                found = True
                break
            elif pair.solid_1 == new_pair.solid_2 and pair.solid_2 == new_pair.solid_1:
                found = True
                break
        if not found:
            d.collision_pairs.append(new_pair)
        else:
            del new_pair

    if constraint.type == 'GENERIC' and (constraint.use_limit_lin_x or constraint.use_limit_lin_y or
                                         constraint.use_limit_lin_z):
        # Check if the user tried to setup transform constraints, warn them these do nothing, if they did.
        print("Warning: Joint '", obj.name, "' has linear-constraints enabled.  "
              "Only angular-constraints are supported.  Ignoring and proceeding.", sep='')

    if constraint.type == 'FIXED':
        min_angles = Vec3([0.0, 0.0, 0.0])
        max_angles = Vec3([0.0, 0.0, 0.0])

    elif constraint.type == 'POINT':
        min_angles = Vec3([-TAU(), -TAU(), -TAU()])
        max_angles = Vec3([TAU(), TAU(), TAU()])

    elif constraint.type == 'HINGE':
        min_angles = Vec3([0.0, 0.0, 0.0])
        max_angles = Vec3([0.0, 0.0, 0.0])
        if constraint.use_limit_ang_z:
            min_angles.z = constraint.limit_ang_z_lower
            max_angles.z = constraint.limit_ang_z_upper

    elif constraint.type == 'GENERIC':
        min_angles = Vec3([0.0, 0.0, 0.0])
        max_angles = Vec3([0.0, 0.0, 0.0])
        if constraint.use_limit_ang_x:
            min_angles.x = constraint.limit_ang_x_lower
            max_angles.x = constraint.limit_ang_x_upper
        if constraint.use_limit_ang_y:
            min_angles.y = constraint.limit_ang_y_lower
            max_angles.y = constraint.limit_ang_y_upper
        if constraint.use_limit_ang_z:
            min_angles.z = constraint.limit_ang_z_lower
            max_angles.z = constraint.limit_ang_z_upper

    else:
        raise SparkException("Constraint type not valid.")

    if obj.parent:
        # not sure what the hell the user is thinking.  Joints shouldn't be parented.
        # I'll just transform back to world-space and proceed as usual with a warning.
        print("Warning: Joint '", obj.name, "' is parented to '", obj.parent.name, "'.  ",
              "This may lead to unintended consequences.  Proceeding.", sep='')

    # local_transform = parent_world_matrix_inverted * object_world_matrix
    matrix_1 = Mat4(constraint.object1.matrix_world.inverted() * obj.matrix_world)
    matrix_2 = Mat4(constraint.object2.matrix_world.inverted() * obj.matrix_world)

    new_joint = Joint()
    new_joint.joint_to_solid_1_coords = Coords(matrix_1)
    new_joint.joint_to_solid_2_coords = Coords(matrix_2)

    # normalize coords
    new_joint.joint_to_solid_1_coords.make_ortho_normal()
    new_joint.joint_to_solid_2_coords.make_ortho_normal()

    new_joint.maximum_angles = max_angles
    new_joint.minimum_angles = min_angles
    new_joint.name = obj.name
    new_joint.solid_1 = get_solid_from_name(d, constraint.object1.name)
    new_joint.solid_2 = get_solid_from_name(d, constraint.object2.name)
    
    if new_joint.solid_1 is None:
        raise SparkException("The object '" + constraint.object1.name + "' -- specified to be Object 1 in the joint '" + obj.name + "' -- does not appear to be a valid solid.")
    if new_joint.solid_2 is None:
        raise SparkException("The object '" + constraint.object2.name + "' -- specified to be Object 2 in the joint '" + obj.name + "' -- does not appear to be a valid solid.")

    return new_joint


def get_world_space_copy(d, solid):
    """
    @type solid: Solid
    """
    parent = solid.bone_name
    mat = Mat4(); mat.from_coords(solid.object_to_bone_coords)
    if parent:
        bone_mat = d.model.armature_object.matrix_world * d.model.bones[d.model.bone_to_index[parent]].matrix_local
        bone_mat = Mat4(bone_mat)
        bone_mat[0][3] *= d.scale_value
        bone_mat[1][3] *= d.scale_value
        bone_mat[2][3] *= d.scale_value
        mat = bone_mat * mat.get_inverse()
    return CollisionMesh(solid, transform=mat)


def get_is_separating_axis(axis: Vec3, mesh1: CollisionMesh, mesh2: CollisionMesh):
    m1_verts_projected = [mesh1.vertices[i].dot_product(axis) for i in range(len(mesh1.vertices))]
    m2_verts_projected = [mesh2.vertices[i].dot_product(axis) for i in range(len(mesh2.vertices))]

    min1 = min(m1_verts_projected)
    min2 = min(m2_verts_projected)

    max1 = max(m1_verts_projected)
    max2 = max(m2_verts_projected)

    return min2 > max1 or min1 > max2


def check_overlap(cm1: CollisionMesh, cm2: CollisionMesh):
    # initialize bound boxes if they're not already
    if not cm1.min_extents:
        cm1.process_bound_box()
    if not cm2.min_extents:
        cm2.process_bound_box()

    # do a quick comparison of bound boxes.  Much much faster but of course not as accurate.  Provides false positives,
    # but not false negatives, so we only use this comparison to prove they AREN'T overlapping.
    if cm1.min_extents.x > cm2.max_extents.x:
        return False
    if cm2.min_extents.x > cm1.max_extents.x:
        return False
    if cm1.min_extents.y > cm2.max_extents.y:
        return False
    if cm2.min_extents.y > cm1.max_extents.y:
        return False
    if cm1.min_extents.z > cm2.max_extents.z:
        return False
    if cm2.min_extents.z > cm1.max_extents.z:
        return False

    # project all vertices of both meshes along the normal vectors of each triangle.  Check for overlap this way.
    for i in range(0, len(cm1.planes)):
        if get_is_separating_axis(cm1.planes[i].normal, cm1, cm2):  # Checks if axis' projection reveals a gap
            return False
    for i in range(0, len(cm2.planes)):
        if get_is_separating_axis(cm2.planes[i].normal, cm1, cm2):  # Checks if axis' projection reveals a gap
            return False

    # check every pair of triangles' cross product as a separation vector
    for i in range(0, len(cm1.planes)):
        for j in range(0, len(cm2.planes)):
            cross_norm, mag = cm1.planes[i].normal.cross_product(cm2.planes[j].normal).normalized_and_mag()
            if mag >= 0.00000001:  # non-zero.  Will be zero if cross product was zero.
                if get_is_separating_axis(cross_norm, cm1, cm2):
                    return False
    return True


def process_collision_pairs(d, rep):
    """
    :type d: ModelData
    :type rep: CollisionRep
    """
    # We automatically disable collision between solids that are interpenetrating in the rest-state.  This can be
    # overridden by the user with the "collisions" directive in the model_compile text.  We'll load up those preferences
    # first.
    if d.read_collision_pairs:  # Will be None unless the user has specified some
        for p in d.read_collision_pairs:
            solid1 = get_solid_from_name(d, p[1])
            solid2 = get_solid_from_name(d, p[2])
            if solid1 and solid2:  # get_solid_from_name returns None if it cannot be found
                if (rep.first_solid_index <= solid1.index < rep.first_solid_index - rep.num_solids and
                   rep.first_solid_index <= solid2.index < rep.first_solid_index - rep.num_solids):
                    new_pair = CollisionPair()
                    new_pair.enabled = p[0]
                    new_pair.solid_1 = solid1
                    new_pair.solid_2 = solid2
                    d.collision_pairs.append(new_pair)

    # Gather up copies of all the solids, and transform them to world-space
    first_index = rep.first_solid_index
    collision_meshes = [get_world_space_copy(d, d.solids[s]) for s in range(first_index, len(d.solids))]
    """:type : list[CollisionMesh]"""
    for i in range(0, len(collision_meshes)):
        for j in range(i + 1, len(collision_meshes)):
            # First, check to see if this pair is already defined
            already_defined = False
            for k in range(len(d.collision_pairs)):
                if ((d.collision_pairs[k].solid_1 == collision_meshes[i].solid_link and
                   d.collision_pairs[k].solid_2 == collision_meshes[j].solid_link) or
                    (d.collision_pairs[k].solid_2 == collision_meshes[i].solid_link and
                   d.collision_pairs[k].solid_1 == collision_meshes[j].solid_link)):
                    already_defined = True
                    break
            if already_defined:
                break
            # Pair is not predefined.  Do a check to see if they overlap
            if check_overlap(collision_meshes[i], collision_meshes[j]):
                new_pair = CollisionPair()
                new_pair.enabled = False
                new_pair.solid_1 = collision_meshes[i].solid_link
                new_pair.solid_2 = collision_meshes[j].solid_link
                d.collision_pairs.append(new_pair)


def read_physics_group(d, rep, physics_group):
    """
    :type d: ModelData
    :type rep: CollisionRep
    :type physics_group: list[str]
    """
    group_index = bpy.data.groups.find(physics_group[1])
    scene_index = bpy.data.scenes.find(physics_group[2])
    if group_index == -1:
        print("Warning!  Physics group '", physics_group[1], "' defined in model_compile block, but not present "
              "in .blend file.  Rep will be empty in exported model.", sep='')
        return
    if scene_index == -1:
        print("Warning!  Physics scene '", physics_group[2], "' defined in model_compile block, but not present "
              "in .blend file.  Rep will be empty in exported model.", sep='')
        return

    group = bpy.data.groups[physics_group[1]]
    scene = bpy.data.scenes[physics_group[2]]

    # objs = [obj for obj in group.objects if obj in scene.objects]  # intersection of group and scene
    objs_group = [obj for obj in group.objects]
    objs_scene = [obj for obj in scene.objects]
    objs = [obj for obj in objs_group if obj in objs_scene]  # intersection of group and scene

    rep.first_solid_index = len(d.solids)
    rep.first_joint_index = len(d.joints)
    rep.first_pair_index = len(d.collision_pairs)

    # read in solids first
    solid_objs = [obj for obj in objs if obj.type == 'MESH']
    if solid_objs:
        orphan_solids = []
        """:type : list[Solid]"""
        for obj in solid_objs:
            new_solid = add_solid(d, scene, obj)
            if new_solid:
                # binary search to figure out where to insert this new solid, the bone indices are sorted ascending
                if not new_solid.bone_name:  # no parent bone, add to separate list for now
                    orphan_solids.append(new_solid)
                else:
                    if not d.solids:
                        d.solids = []
                        d.solids.append(new_solid)
                        continue
                    new_bone_index = d.model.bone_to_index[new_solid.bone_name]
                    right_index = len(d.solids) - 1
                    if d.model.bone_to_index[d.solids[right_index].bone_name] < new_bone_index:  # add to end
                        d.solids.append(new_solid)
                    else:
                        left_index = 0
                        middle_index = (right_index + left_index) // 2  # '//' means integer division, round down
                        while right_index > left_index:
                            middle_value = d.model.bone_to_index[d.solids[middle_index].bone_name]
                            if new_bone_index > middle_value:
                                left_index = middle_index + 1
                            elif new_bone_index < middle_value:
                                right_index = middle_index
                            else:  # Only possible outcome at this point is the bone indices are equal
                                raise SparkException("Error!  Bone '" + new_solid.bone_name + "' is referenced by "
                                                     "multiple Solids.  This is not acceptable, as the solid drives "
                                                     "the bone during rag-doll.")
                            middle_index = (right_index + left_index) // 2  # '//' means integer division, round down
                        d.solids.insert(middle_index, new_solid)
                        for i in range(middle_index + 1, len(d.solids)):
                            d.solids[i].index += 1
        for o in orphan_solids:  # add them now.  They would've complicated the sorting process a bit
            d.solids.append(o)
        for i in range(len(d.solids)):
            d.solids[i].index = i
        solid_names = [s.name for s in d.solids]

        # read in joints
        for obj in objs:
            if obj.type == 'EMPTY':
                new_joint = add_joint(d, obj, solid_names)
                if new_joint:
                    new_joint.index = len(d.joints)
                    d.joints.append(new_joint)

        # process collision pairs
        process_collision_pairs(d, rep)

        # spit out errors for non mesh, empty objects, just to let the user know
        # that one of their objects isn't being processed.
        for obj in objs:
            if obj.type != 'EMPTY' and obj.type != 'MESH':
                print("Warning: Object '", obj.name, "' is not valid to be a member of this physics group.  Objects "
                      "must be of type 'EMPTY' or 'MESH' to be a joint or a solid, respectively.")

        rep.num_solids = len(d.solids) - rep.first_solid_index
        rep.num_joints = len(d.joints) - rep.first_joint_index
        rep.num_pairs = len(d.collision_pairs) - rep.first_pair_index

    else:
        print("Warning!  No MESH-type objects present for rep '", physics_group[0], "'.  Rep will be empty in "
              "exported model.")
        rep.num_solids = 0
        rep.num_joints = 0
        rep.num_pairs = 0


# Load all the physics data specified in the model_compile text block
def load_physics(d):
    reps = d.collision_reps
    rep_entries = d.collision_rep_entries

    if not d.physics_groups:  # if none or empty, skip physics
        return

    # load up default first
    new_rep = CollisionRep()
    reps.append(new_rep)
    rep_entries.append(CollisionRepEntry())
    rep_entries[0].name = 'default'
    rep_entries[0].collision_rep_index = 0

    default_index = -1
    for i in range(0, len(d.physics_groups)):
        if d.physics_groups[i][0] == 'default':
            default_index = i
            break
    if default_index >= 0:  # default exists, read it in
        read_physics_group(d, new_rep, d.physics_groups[default_index])

    # load up the others now
    for i in range(0, len(d.physics_groups)):
        if i == default_index:
            continue  # skip, as it would have already been read-in.
        new_rep_entry = CollisionRepEntry()
        rep_entries.append(new_rep_entry)
        new_rep_entry.name = d.physics_groups[i][0]
        new_rep_entry.collision_rep_index = len(reps)
        new_rep = CollisionRep()
        reps.append(new_rep)
        read_physics_group(d, new_rep, d.physics_groups[i])
