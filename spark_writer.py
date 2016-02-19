# 1455664595
# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

from struct import *
from spark_common import *
from spark_animation import *


class SparkWriter:
    def __init__(self):
        self.data = bytearray(0)
        self.pos = 0
        self.chunk_length_pos = []  # stack of positions with 4 bytes reserved for the length field

    def allocate(self, n):
        self.data.extend(b'\x00' * n)

    def ensure_bytes(self, n):
        if self.pos + n > len(self.data):
            self.allocate(n + self.pos - len(self.data))

    def write_raw(self, val):
        self.ensure_bytes(len(val))
        self.data[self.pos:self.pos + len(val)] = val
        self.pos += len(val)

    def begin_chunk(self, chunk_id):
        self.write_int32(CHUNK_ID(chunk_id) if isinstance(chunk_id, str) else chunk_id)
        self.chunk_length_pos.append(self.pos)
        self.write_int32(0)  # dummy value for length

    def end_chunk(self):
        end_pos = self.pos
        self.pos = self.chunk_length_pos.pop()
        self.write_int32(end_pos - self.pos - 4)
        self.pos = end_pos

    def write_int32(self, val):
        self.ensure_bytes(4)
        self.data[self.pos:self.pos + 4] = pack("<L", val)
        self.pos += 4

    def write_string(self, s_val):
        self.write_int32(len(s_val))
        self.ensure_bytes(len(s_val))
        self.data[self.pos:self.pos + len(s_val)] = s_val.encode()
        self.pos += len(s_val)

    def write_float(self, val):
        self.ensure_bytes(4)
        self.data[self.pos:self.pos + 4] = pack("<f", val)
        self.pos += 4

    def write_vec3(self, vec):
        self.ensure_bytes(12)
        self.data[self.pos:self.pos + 12] = pack("<fff", vec[0], vec[1], vec[2])
        self.pos += 12

    def write_quat(self, qt):
        self.ensure_bytes(16)
        self.data[self.pos:self.pos + 16] = pack("<ffff", qt.x, qt.y, qt.z, qt.w)
        self.pos += 16

    def write_coords(self, c: Coords):
        self.ensure_bytes(48)
        self.data[self.pos:self.pos + 48] = pack("<ffffffffffff", c.x_axis.x, c.x_axis.y, c.x_axis.z,
                                                                  c.y_axis.x, c.y_axis.y, c.y_axis.z,
                                                                  c.z_axis.x, c.z_axis.y, c.z_axis.z,
                                                                  c.origin.x, c.origin.y, c.origin.z)
        self.pos += 48

    def write_affine_parts(self, a: AffineParts):
        self.ensure_bytes(60)
        self.data[self.pos:self.pos + 60] = pack("<fffffffffffffff",
                                                 a.translation.x, a.translation.y, a.translation.z,
                                                 a.rotation.x, a.rotation.y, a.rotation.z, a.rotation.w,
                                                 a.scale.x, a.scale.y, a.scale.z,
                                                 a.scale_rotation.x, a.scale_rotation.y,
                                                 a.scale_rotation.z, a.scale_rotation.w,
                                                 a.flip)
        self.pos += 60

    def write_bool(self, val):
        self.ensure_bytes(4)
        self.data[self.pos:self.pos + 4] = pack("<L", 1 if val else 0)
        self.pos += 4

    def write_vertex(self, v: Vertex, bone_offset=0):
        self.ensure_bytes(92)
        self.data[self.pos:self.pos + 92] = pack("<ffffffffffffffLfLfLfLfL",
                                             v.co[0], v.co[1], v.co[2],
                                             v.nrm[0], v.nrm[1], v.nrm[2],
                                             v.tan[0], v.tan[1], v.tan[2],
                                             v.bin[0], v.bin[1], v.bin[2],
                                             v.t_co[0], v.t_co[1], 0xFFFFFFFF,
                                             v.bone_weights[0].weight if v.bone_weights else 1.0,
                                             v.bone_weights[0].index + bone_offset if v.bone_weights else 0,
                                             v.bone_weights[1].weight if len(v.bone_weights) > 1 else 0.0,
                                             v.bone_weights[1].index + bone_offset if len(v.bone_weights) > 1 else 0,
                                             v.bone_weights[2].weight if len(v.bone_weights) > 2 else 0.0,
                                             v.bone_weights[2].index + bone_offset if len(v.bone_weights) > 2 else 0,
                                             v.bone_weights[3].weight if len(v.bone_weights) > 3 else 0.0,
                                             v.bone_weights[3].index + bone_offset if len(v.bone_weights) > 3 else 0)
        self.pos += 92

    def close_and_return(self):
        return self.data[0:self.pos]


