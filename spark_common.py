# 1455664595
# Blender -> Spark .model exporter
# Natural Selection 2 model compile utility written
# by Max McGuire and Steve An of Unknown Worlds Entertainment
# Adapted to Python for Blender by Trevor "BeigeAlert" Harris

# Spark-related utilities

import math
import mathutils


# noinspection PyPep8Naming
def TAU(): return math.pi * 2


# noinspection PyPep8Naming
# In retrospect, a dict would have been awesome here... :/
def CHUNK_ID(name):
    if name == 'Chunk_Vertices':
        return 1
    if name == 'Chunk_Indices':
        return 2
    if name == 'Chunk_FaceSets':
        return 3
    if name == 'Chunk_Materials':
        return 4
    if name == 'Chunk_Solids':
        return 5
    if name == 'Chunk_Bones':
        return 6
    if name == 'Chunk_Animations':
        return 7
    if name == 'Chunk_AnimationNodes':
        return 8
    if name == 'Chunk_Sequences':
        return 9
    if name == 'Chunk_BlendParameters':
        return 10
    if name == 'Chunk_Cameras':
        return 11
    if name == 'Chunk_HitProxies':  # Never used, but good to have its legacy documented.
        return 12
    if name == 'Chunk_AttachPoints':
        return 13
    if name == 'Chunk_Joints':
        return 14
    if name == 'Chunk_CollisionPairs':
        return 15
    if name == 'Chunk_CollisionReps':
        return 16
    if name == 'Chunk_BoundingBox':
        return 17
    if name == 'Chunk_BoneBoundingBoxes':
        return 18
    if name == 'Chunk_AnimationModel':
        return 19
    raise SparkException("Invalid chunk identifier '" + name + "'.")


class SparkException(Exception):
    pass


def vert_x(vert): return vert[0]


def vert_y(vert): return vert[1]


def vert_z(vert): return vert[2]


def cross_product(v1, v2):
    v = v1
    w = v2
    if isinstance(v, Vec3):
        v = v()
    if isinstance(w, Vec3):
        w = w()
    
    vect = Vec3()
    vect()[0] = v[1] * w[2] - v[2] * w[1]
    vect()[1] = v[2] * w[0] - v[0] * w[2]
    vect()[2] = v[0] * w[1] - v[1] * w[0]
    return vect


class MinMaxVec3:  # Used to construct bounding boxes for lots of data, eg. vertex coordinates
    def __init__(self, *args):
        self.minimum = None
        self.maximum = None
        if len(args) == 1:
            self.minimum = args[0]
            self.maximum = args[0]
    
    def min_max(self, vec):  # Add point to bound box
        if vec is None:
            raise SparkException("Expected a Vec3, got None type!")
        if self.minimum is None:
            self.minimum = vec
            self.maximum = vec
        else:
            self.minimum = [min(self.minimum[i], vec[i]) for i in range(3)]
            self.maximum = [max(self.maximum[i], vec[i]) for i in range(3)]

    def merge(self, mx):  # merge this minmaxvec3 with another
        """
        :type mx : MinMaxVec3
        """
        if self.minimum is not None:
            if mx.minimum is not None:
                self.minimum = [min(self.minimum[i], mx.minimum[i]) for i in range(3)]
                self.maximum = [max(self.maximum[i], mx.maximum[i]) for i in range(3)]
        else:
            self.minimum = mx.minimum
            self.maximum = mx.maximum


class BoundBox:
    def __init__(self, *args):
        self.origin = [0.0, 0.0, 0.0]
        self.extents = [0.0, 0.0, 0.0]
        if len(args) == 1:
            # assume arg is a MinMaxVec3
            if args[0].minimum is not None:
                self.origin = [(args[0].minimum[i] + args[0].maximum[i]) / 2.0 for i in range(3)]
                self.extents = [args[0].maximum[i] - self.origin[i] for i in range(3)]
    
    def get_min(self):
        return [self.origin[0] - self.extents[0], self.origin[1] - self.extents[1], self.origin[2] - self.extents[2]]
    
    def get_max(self):
        return [self.origin[0] + self.extents[0], self.origin[1] + self.extents[1], self.origin[2] + self.extents[2]]
    
    def get_verts(self):  # returns 8 Vec3's that form the bounding box volume
        minimum = self.get_min()
        maximum = self.get_max()

        return [Vec3([maximum[0], maximum[1], maximum[2]]),
                Vec3([maximum[0], maximum[1], minimum[2]]),
                Vec3([maximum[0], minimum[1], maximum[2]]),
                Vec3([maximum[0], minimum[1], minimum[2]]),
                Vec3([minimum[0], maximum[1], maximum[2]]),
                Vec3([minimum[0], maximum[1], minimum[2]]),
                Vec3([minimum[0], minimum[1], maximum[2]]),
                Vec3([minimum[0], minimum[1], minimum[2]])]

    def from_verts(self, verts):
        min_val = [min(verts, key=vert_x)[0], min(verts, key=vert_y)[1], min(verts, key=vert_z)[2]]
        max_val = [max(verts, key=vert_x)[0], max(verts, key=vert_y)[1], max(verts, key=vert_z)[2]]
        
        self.origin = [(min_val[0] + max_val[0]) / 2.0, (min_val[1] + max_val[1]) / 2.0,
                       (min_val[2] + max_val[2]) / 2.0]
        self.extents = [self.origin[i] - min_val[i] for i in range(3)]


class Mat3:
    def __init__(self):
        self.data = None
        self.set_to_identity()
    
    def __call__(self, *args):
        if len(args) == 0:
            return self.data
        elif len(args) == 2:
            return self.data[args[0]][args[1]]
    
    def __sub__(self, other):
        result = Mat3()
        for r in range(0, 3):
            for c in range(0, 3):
                result()[r][c] = self(r, c) - other(r, c)
        return result
    
    def __mul__(self, other):
        result = Mat3()
        for i in range(0, 3):
            for j in range(0, 3):
                cell_sum = 0.0
                for k in range(0, 3):
                    cell_sum += self(i, k) * other(k, j)
                
                result()[i][j] = cell_sum
        return result
    
    def __neg__(self):
        result = Mat3()
        for i in range(0, 3):
            for j in range(0, 3):
                result()[i][j] = -self(i, j)
        return result

    def transpose(self):
        self.data[0][1], self.data[1][0] = self.data[1][0], self.data[0][1]
        self.data[0][2], self.data[2][0] = self.data[2][0], self.data[0][2]
        self.data[1][2], self.data[2][1] = self.data[2][1], self.data[1][2]

    def get_row(self, r):
        return self.data[r]

    def get_copy(self):
        copy = Mat3()
        for r in range(0, 3):
            for c in range(0, 3):
                copy()[r][c] = self(r, c)
        return copy

    def set_row(self, r, vect):
        self.data[r] = vect()[:]

    def set_to_identity(self):
        self.data = [[1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]]

    def get_max_column(self):
        col = -1
        maximum = 0.0
        for r in range(0, 3):
            for c in range(0, 3):
                absolute = abs(self(r, c))
                if absolute > maximum:
                    maximum = absolute
                    col = c
        return col

    def get_transpose(self):
        matrix = Mat3()
        for r in range(0, 3):
            for c in range(0, 3):
                matrix.data[r][c] = self.data[c][r]
        return matrix

    def get_determinant(self):
        m = self.data
        return ((m[0][0] * m[1][1] * m[2][2]) - (m[2][0] * m[1][1] * m[0][2]) +
                (m[0][1] * m[1][2] * m[2][0]) - (m[0][1] * m[1][0] * m[2][2]) +
                (m[1][0] * m[2][1] * m[0][2]) - (m[1][2] * m[2][1] * m[0][0]))

    def mat_norm(self, tpose):
        maximum = 0.0
        for i in range(0, 3):
            if tpose:
                m_sum = abs(self(0, i)) + abs(self(1, i)) + abs(self(2, i))
            else:
                m_sum = abs(self(i, 0)) + abs(self(i, 1)) + abs(self(i, 2))
            if maximum < m_sum:
                maximum = m_sum
        return maximum

    def norm_inf(self):
        return self.mat_norm(0)

    def norm_one(self):
        return self.mat_norm(1)

    def get_adjoint_transpose(self):
        matrix = Mat3()
        matrix.set_row(0, cross_product(self.get_row(1), self.get_row(2)))
        matrix.set_row(1, cross_product(self.get_row(2), self.get_row(0)))
        matrix.set_row(2, cross_product(self.get_row(0), self.get_row(1)))
        return matrix

    def reflect_columns(self, u):
        for i in range(0, 3):
            s = u(0) * self(0, i) + u(1) * self(1, i) + u(2) * self(2, i)
            
            for j in range(0, 3):
                self()[j][i] = self(j, i) - u(j) * s

    def reflect_rows(self, u):
        for i in range(0, 3):
            s = u(0) * self(i, 0) + u(1) * self(i, 1) + u(2) * self(i, 2)
            
            for j in range(0, 3):
                self()[i][j] = self(i, j) - u(j) * s

    @staticmethod
    def make_reflector(v):
        u = Vec3()
        s = v.get_length()
        
        u.x = v(0)
        u.y = v(1)
        u.z = v(2) + (-s if v(2) < 0.0 else s)
        
        s = math.sqrt(2.0 / u.get_length_squared())
        
        u.xyz = [u(i) * s for i in range(0, 3)]
        
        return u

    def do_rank_1(self):
        q = Mat3()  # identity matrix
        col = self.get_max_column()
        
        if col < 0:
            return q
        
        v1 = Vec3()
        v1.xyz = [self(i, col) for i in range(0, 3)]
        v1 = Mat3.make_reflector(v1)
        self.reflect_columns(v1)
        
        v2 = Vec3()
        v2.xyz = [self(2, i) for i in range(0, 3)]
        v2 = Mat3.make_reflector(v2)
        self.reflect_rows(v2)
        
        s = self(2, 2)
        
        if s < 0.0:
            q()[2][2] = -1.0
        
        q.reflect_columns(v1)
        q.reflect_rows(v2)
        return q

    def do_rank_2(self, madjt):
        q = self.get_copy()
        col = madjt.get_max_column()
        
        if col < 0:
            return self.do_rank_1()
        
        v1 = Vec3()
        v1.xyz = [madjt(0, i) for i in range(0, 3)]
        
        v1 = Mat3.make_reflector(v1)
        self.reflect_columns(v1)
        
        v2 = Vec3()
        v2.xyz = cross_product(self.get_row(0), self.get_row(1))
        v2 = Mat3.make_reflector(v2)
        self.reflect_rows(v2)
        
        w = self(0, 0)
        x = self(0, 1)
        y = self(1, 0)
        z = self(1, 1)
        
        if w * z > x * y:
            c = z + w
            s = y - x
            d = math.sqrt(c * c + s * s)

            c /= d
            s /= d
            
            q()[0][0] = c
            q()[1][1] = c
            q()[0][1] = -s
            q()[1][0] = s
        else:
            c = z - w
            s = y + x
            d = math.sqrt(c * c + s * s)

            c /= d
            s /= d
            
            q()[0][0] = -c
            q()[1][1] = c
            q()[0][1] = s
            q()[1][0] = s
        
        q()[0][2] = 0.0
        q()[2][0] = 0.0
        q()[1][2] = 0.0
        q()[2][1] = 0.0
        q()[2][2] = 1.0
        
        q.reflect_columns(v1)
        q.reflect_rows(v2)
        return q

    def get_polar_decompose(self):
        tolerance = 0.000001
        
        mk = self.get_transpose()
        m_one = mk.norm_one()
        m_inf = mk.norm_inf()

        det = 0.0
        
        while True:
            madjtk = mk.get_adjoint_transpose()
            det = (mk(0, 0) * madjtk(0, 0)) + (mk(0, 1) * madjtk(0, 1)) + (mk(0, 2) * madjtk(0, 2))
            
            if det == 0.0:
                mk = mk.do_rank_2(madjtk)
                break
            
            madjt_one = madjtk.norm_one()
            madjt_inf = madjtk.norm_inf()
            gamma = math.sqrt(math.sqrt((madjt_one * madjt_inf) / (m_one * m_inf)) / abs(det))
            g1 = gamma * 0.5
            g2 = 0.5 / (gamma * det)
            
            ek = mk.get_copy()  # copy, not reference
            
            for i in range(0, 3):
                for j in range(0, 3):
                    mk()[i][j] = g1 * mk(i, j) + g2 * madjtk(i, j)
            
            ek = ek - mk
            
            e_one = ek.norm_one()
            m_one = mk.norm_one()
            m_inf = mk.norm_inf()
            
            # loop terminator
            if not (e_one > (m_one * tolerance)):
                break
        
        q = mk.get_transpose()
        s = mk * self
        
        for i in range(0, 3):
            for j in range(0, 3):
                s()[i][j] = s()[j][i] = 0.5 * (s(i, j) + s(j, i))
        
        return q, s, det

    def get_spectral_decompose(self):
        u = Mat3()
        diag = [self(i, i) for i in range(0, 3)]
        offd = [self((i + 1) % 3, (i + 2) % 3) for i in range(0, 3)]
        
        for sweep in range(20, 0, -1):  # start @20, count down 1 until at 0
            sm = abs(offd[0]) + abs(offd[1]) + abs(offd[2])
            
            if sm == 0.0:
                break
            
            for i in range(2, -1, -1):  # 2,1,0
                p = (i + 1) % 3
                q = (p + 1) % 3
                
                fabsoffdi = abs(offd[i])
                g = 100.0 * fabsoffdi
                
                if fabsoffdi > 0.0:
                    h = diag[q] - diag[p]
                    fabsh = abs(h)

                    if fabsh + g == fabsh:
                        t = offd[i] / h
                    else:
                        theta = 0.5 * h / offd[i]
                        t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                        if theta < 0.0:
                            t = -t

                    c = 1.0 / math.sqrt(t * t + 1.0)

                    s = t * c
                    tau = s / (c + 1.0)
                    ta = t * offd[i]

                    offd[i] = 0.0
                    diag[p] -= ta
                    diag[q] += ta

                    offdq = offd[q]
                    offd[q] -= s * (offd[p] + tau * offd[q])
                    offd[p] += s * (offdq - tau * offd[p])

                    for j in range(2, -1, -1):
                        a = u(j, p)
                        b = u(j, q)
                        u()[j][p] -= s * (b + tau * a)
                        u()[j][q] += s * (a - tau * b)
                
        kv = Vec3()
        kv.xyz = [diag[i] for i in range(0, 3)]
        
        return kv, u
        

class Mat4:
    def __init__(self, *args):
        self.data = [[1.0, 0.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0, 0.0],
                     [0.0, 0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0, 1.0]]
        if len(args) == 1:
            if isinstance(args[0], mathutils.Matrix):
                self.from_blender(args[0])
            elif isinstance(args[0], Mat4):
                for r in range(4):
                    for c in range(4):
                        self.data[r][c] = args[0][r][c]

    def __call__(self, *args):
        if len(args) == 0:
            return self.data
        elif len(args) == 2:
            return self.data[args[0]][args[1]]

    def __setattr__(self, name, value):
        if name == 'm00': self.data[0][0] = value
        elif name == 'm01': self.data[0][1] = value
        elif name == 'm02': self.data[0][2] = value
        elif name == 'm03': self.data[0][3] = value
        elif name == 'm10': self.data[1][0] = value
        elif name == 'm11': self.data[1][1] = value
        elif name == 'm12': self.data[1][2] = value
        elif name == 'm13': self.data[1][3] = value
        elif name == 'm20': self.data[2][0] = value
        elif name == 'm21': self.data[2][1] = value
        elif name == 'm22': self.data[2][2] = value
        elif name == 'm23': self.data[2][3] = value
        elif name == 'm30': self.data[3][0] = value
        elif name == 'm31': self.data[3][1] = value
        elif name == 'm32': self.data[3][2] = value
        elif name == 'm33': self.data[3][3] = value
        else: self.__dict__[name] = value

    def __getattr__(self, name):
        if name == 'm00': return self.data[0][0]
        elif name == 'm01': return self.data[0][1]
        elif name == 'm02': return self.data[0][2]
        elif name == 'm03': return self.data[0][3]
        elif name == 'm10': return self.data[1][0]
        elif name == 'm11': return self.data[1][1]
        elif name == 'm12': return self.data[1][2]
        elif name == 'm13': return self.data[1][3]
        elif name == 'm20': return self.data[2][0]
        elif name == 'm21': return self.data[2][1]
        elif name == 'm22': return self.data[2][2]
        elif name == 'm23': return self.data[2][3]
        elif name == 'm30': return self.data[3][0]
        elif name == 'm31': return self.data[3][1]
        elif name == 'm32': return self.data[3][2]
        elif name == 'm33': return self.data[3][3]
        else: return None

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __mul__(self, other):
        if isinstance(other, Mat4):  # matrix multiplication
            result = Mat4()
            for i in range(0, 4):
                for j in range(0, 4):
                    cell_sum = 0.0
                    for k in range(0, 4):
                        cell_sum += self(i, k) * other(k, j)
                    result[i][j] = cell_sum
            return result
        elif isinstance(other, Vec3):  # vector transformation
            result = Vec3()
            other_data = [other.x, other.y, other.z, 1]
            for i in range(0, 3):
                cell_sum = 0.0
                for j in range(0, 4):
                    cell_sum += self(i, j) * other_data[j]
                result[i] = cell_sum
            return result
        else:
            raise TypeError('Mat4 can only be multiplied by Mat4 or Vec3.')
                        
    def fix_axes(self, reverse=False):
        if reverse:
            # y from x, z from y, and x from z (that is, Blender from Spark)
            self.data[0], self.data[1], self.data[2] = self.data[1], self.data[2], self.data[0]
        else:
            # x from y, y from z, and z from x (that is, Spark from Blender)
            self.data[0], self.data[1], self.data[2] = self.data[2], self.data[0], self.data[1]
    
    def from_blender(self, mat, axis_fix=False, reverse=False):
        for r in range(4):
            for c in range(4):
                self.data[r][c] = mat[r][c]
                
        if axis_fix:
            self.fix_axes(reverse=reverse)
    
    def to_blender(self):
        mat = mathutils.Matrix()
        for r in range(4):
            for c in range(4):
                mat[r][c] = self.data[r][c]
        return mat

    def from_coords(self, coords):
        """
        :type coords: Coords
        """
        self[0][0] = coords.x_axis.x
        self[1][0] = coords.x_axis.y
        self[2][0] = coords.x_axis.z
        self[3][0] = 0.0

        self[0][1] = coords.y_axis.x
        self[1][1] = coords.y_axis.y
        self[2][1] = coords.y_axis.z
        self[3][1] = 0.0

        self[0][2] = coords.z_axis.x
        self[1][2] = coords.z_axis.y
        self[2][2] = coords.z_axis.z
        self[3][2] = 0.0

        self[0][3] = coords.origin.x
        self[1][3] = coords.origin.y
        self[2][3] = coords.origin.z
        self[3][3] = 1.0

    def transpose(self):
        for c in range(0, 3):
            for r in range(c + 1, 4):
                self.data[c][r], self.data[r][c] = self.data[r][c], self.data[c][r]

    def get_transpose(self):
        matrix = Mat4()
        for r in range(0, 4):
            for c in range(0, 4):
                matrix.data[r][c] = self.data[c][r]
        return matrix

    @staticmethod
    def get_scale_matrix(scale):
        matrix = Mat4()
        matrix.data = [[scale, 0.0, 0.0, 0.0],
                       [0.0, scale, 0.0, 0.0],
                       [0.0, 0.0, scale, 0.0],
                       [0.0, 0.0, 0.0, 1.0]]
        return matrix

    def get_sub_matrix(self, row, col):  # create a Mat3 by excluding the specified row and column
        matrix = Mat3()
        for r in range(0, 3):
            for c in range(0, 3):
                matrix.data[r][c] = self.data[r if r < row else r + 1][c if c < col else c + 1]
        return matrix

    def get_determinant(self):
        result = 0.0
        for i in range(0, 4):
            result += self.data[0][i] * self.get_sub_matrix(0, i).get_determinant() * (1.0 if (i & 1) == 0 else -1.0)
        return result

    def get_adjoint(self):
        matrix = Mat4()
        for r in range(0, 4):
            for c in range(0, 4):
                matrix.data[c][r] = self.get_sub_matrix(r, c).get_determinant() * (1.0 if ((r + c) & 1) == 0 else -1.0)
        return matrix

    def get_inverse(self):
        matrix = Mat4()
        adjoint = self.get_adjoint()
        
        det = 0.0
        for i in range(0, 4):
            det += self.data[0][i] * adjoint[i][0]
        
        for r in range(0, 4):
            for c in range(0, 4):
                matrix[r][c] = adjoint[r][c] / det
        return matrix


class Coords:
    def __init__(self, *args):
        self.x_axis = Vec3([1.0, 0.0, 0.0])
        self.y_axis = Vec3([0.0, 1.0, 0.0])
        self.z_axis = Vec3([0.0, 0.0, 1.0])
        self.origin = Vec3([0.0, 0.0, 0.0])
        if len(args) == 1:
            if isinstance(args[0], Mat4):
                self.x_axis = Vec3([args[0][0][0], args[0][1][0], args[0][2][0]])
                self.y_axis = Vec3([args[0][0][1], args[0][1][1], args[0][2][1]])
                self.z_axis = Vec3([args[0][0][2], args[0][1][2], args[0][2][2]])
                self.origin = Vec3([args[0][0][3], args[0][1][3], args[0][2][3]])

    @staticmethod
    def get_identity():
        co = Coords()
        co.x_axis = Vec3([1.0, 0.0, 0.0])
        co.y_axis = Vec3([0.0, 1.0, 0.0])
        co.z_axis = Vec3([0.0, 0.0, 1.0])
        co.origin = Vec3([0.0, 0.0, 0.0])
        return co

    def make_ortho_normal(self):
        scale = Vec3()
        scale.x = self.x_axis.get_length()
        scale.y = self.y_axis.get_length()
        scale.z = self.z_axis.get_length()

        self.x_axis *= 1.0 / scale.x
        self.y_axis *= 1.0 / scale.y
        self.z_axis *= 1.0 / scale.z

        mirror = self.x_axis.cross_product(self.y_axis).dot_product(self.z_axis)
        if mirror < 0.0:
            self.z_axis *= -1.0

        self.z_axis = self.x_axis.cross_product(self.y_axis).normalized()
        self.y_axis = self.z_axis.cross_product(self.x_axis).normalized()


class Vec3:
    def __init__(self, *args, fix_axes=False, reverse=False):
        self.data = [0.0, 0.0, 0.0]
        if len(args) == 1:  # length-3 list of floats, or a vec3
            self.data = args[0][:]
            if fix_axes:
                if reverse:
                    # y from x, z from y, and x from z (that is, Blender from Spark)
                    self.data[0], self.data[1], self.data[2] = self.data[1], self.data[2], self.data[0]
                else:
                    # x from y, y from z, and z from x (that is, Spark from Blender)
                    self.data[0], self.data[1], self.data[2] = self.data[2], self.data[0], self.data[1]

    def __call__(self, *args):
        if len(args) == 0:
            return self.data
        elif len(args) == 1:
            return self.data[args[0]]

    def __getattr__(self, name):
        if name == 'x': return self.data[0]
        elif name == 'y': return self.data[1]
        elif name == 'z': return self.data[2]
        elif name == 'xyz': return [self.data[0], self.data[1], self.data[2]]
        else: return None
        
    def __setattr__(self, name, value):
        if name == 'x': self.data[0] = value
        elif name == 'y': self.data[1] = value
        elif name == 'z': self.data[2] = value
        elif name == 'xyz': self.data = value
        else: self.__dict__[name] = value
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __sub__(self, other):
        if isinstance(other, Vec3):
            result = Vec3()
            result.x = self.x - other.x
            result.y = self.y - other.y
            result.z = self.z - other.z
            return result
        else:
            raise TypeError("Vec3 can only be subtracted from another Vec3.")

    def __add__(self, other):
        if isinstance(other, Vec3):
            result = Vec3()
            result.x = self.x + other.x
            result.y = self.y + other.y
            result.z = self.z + other.z
            return result
        else:
            raise TypeError("Vec3 can only be added to another Vec3.")

    def __mul__(self, other):
        try:
            x = float(other)
            result = Vec3(self)
            result[0] *= x; result[1] *= x; result[2] *= x
            return result
        except:
            raise TypeError("Vec3 can only be multiplied by a scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def get_length_squared(self):
        return self(0) * self(0) + self(1) * self(1) + self(2) * self(2)
    
    def get_length(self):
        return math.sqrt(self.get_length_squared())

    def dot_product(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross_product(self, other):
        new_vec = Vec3()
        new_vec.x = self.y * other.z - self.z * other.y
        new_vec.y = self.z * other.x - self.x * other.z
        new_vec.z = self.x * other.y - self.y * other.x
        return new_vec

    def normalized(self):
        new_vec = Vec3()
        mag = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if mag <= 0.00000001:  # zero
            new_vec.x = 0.0
            new_vec.y = 0.0
            new_vec.z = 0.0
            return new_vec
        new_vec.x = self.x / mag
        new_vec.y = self.y / mag
        new_vec.z = self.z / mag
        return new_vec

    def normalized_and_mag(self):  # same as above, but returns the magnitude as well as the normalized vector
        new_vec = Vec3()
        mag = math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        if mag <= 0.00000001:  # zero
            new_vec.x = 0.0
            new_vec.y = 0.0
            new_vec.z = 0.0
            return new_vec, 0.0
        new_vec.x = self.x / mag
        new_vec.y = self.y / mag
        new_vec.z = self.z / mag
        return new_vec, mag


class Quat:
    def __init__(self, *args):
        self.data = [0.0, 0.0, 0.0, 0.0]  # wxyz
        if len(args) == 1:  # copy of existing quat
            self.data = args[0].data[:]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, item):
        return self.data[item]

    def __getattr__(self, name):
        if name == 'x': return self.data[1]
        elif name == 'y': return self.data[2]
        elif name == 'z': return self.data[3]
        elif name == 'w': return self.data[0]
        elif name == 'xyz': return [self.data[1], self.data[2], self.data[3]]
        elif name == 'wxyz': return [self.data[:]]
        else: return None

    def __setattr__(self, name, value):
        if name == 'x': self.data[1] = value
        elif name == 'y': self.data[2] = value
        elif name == 'z': self.data[3] = value
        elif name == 'w': self.data[0] = value
        elif name == 'wxyz': self.data = value
        else: self.__dict__[name] = value
    
    def __mul__(self, other):
        result = Quat()
        result.x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        result.y = self.w * other.y + self.y * other.w + self.z * other.x - self.x * other.z
        result.z = self.w * other.z + self.z * other.w + self.x * other.y - self.y * other.x
        result.w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        return result
    
    def dot_product(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    
    def get_conjugate(self):
        result = Quat()
        result.w = self.w
        result.x = -self.x
        result.y = -self.y
        result.z = -self.z
        return result

    # noinspection PyAttributeOutsideInit
    def set_from_matrix(self, m):
        f_trace = m(0, 0) + m(1, 1) + m(2, 2)
        
        if f_trace > 0.0:
            f_root = math.sqrt(f_trace + 1.0)
            self.w = 0.5 * f_root
            f_root = 0.5 / f_root
            self.x = (m(2, 1) - m(1, 2)) * f_root
            self.y = (m(0, 2) - m(2, 0)) * f_root
            self.z = (m(1, 0) - m(0, 1)) * f_root
        else:
            i = 0
            if m(1, 1) > m(0, 0):
                i = 1
            if m(2, 2) > m(i, i):
                i = 2
            j = (i + 1) % 3
            k = (j + 1) % 3
            
            f_root = math.sqrt(m(i, i) - m(j, j) - m(k, k) + 1.0)
            self[i + 1] = 0.5 * f_root
            f_root = 0.5 / f_root
            self.w = (m(k, j) - m(j, k)) * f_root
            self[j + 1] = (m(j, i) + m(i, j)) * f_root
            self[k + 1] = (m(k, i) + m(i, k)) * f_root
    
    def get_distance(self, other):
        return math.sqrt(self.get_distance_squared(other))

    def get_distance_squared(self, other):
        sign = 1.0 if self.dot_product(other) >= 0 else -1.0
        
        dx = self.x - sign * other.x
        dy = self.y - sign * other.y
        dz = self.z - sign * other.z
        dw = self.w - sign * other.w
        return dx * dx + dy * dy + dz * dz + dw * dw


