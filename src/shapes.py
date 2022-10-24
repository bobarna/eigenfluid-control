'''
Generating and sampling shapes

u,v \in [0,1]
'''

# TODO Derive each class from PhiFlow Geometry class

from phi.torch.flow import *

# Helper function to sample initial and target position of 2 shapes
# Get O overlapping, and U non-necessarily overlapping ('unique') sample points
def get_points_for_shapes(shape_0, shape_target, O=30, U=30):
    sampler_union = ShapeSampler(shape_0, shape_target, N=O, h1=2, h2=7)
    sampler_0 = ShapeSampler(shape_0, N=U, h1=3, h2=11)
    sampler_target = ShapeSampler(shape_target, N=U, h1=3, h2=11)
    # Concatenate both the union and non-union points
    p_0 = math.concat((sampler_union.p, sampler_0.p), instance('i'))
    p_0 = shape_0.create_points(p_0)
    p_t = math.concat((sampler_union.p, sampler_target.p), instance('i'))
    p_t = shape_target.create_points(p_t)

    return (p_0, p_t)

'''
N: number of sample points
h1, h2: primes for halton sequence
A, B: shapes, of which the subsection is sampled
    if B is None, then only A is sampled
'''
class ShapeSampler:
    def __init__(self, A, B=None, N=16, h1=5, h2=3):
        self.A = A # shape 1 
        self.B = B # shape 2
        self.N = N
        self.h1 = h1
        self.h2 = h2

        self.p = []

        # generate N points in [0,1]x[0,1]
        # Check wheter (u,v) is inside the shape
        # if not, then throw it away, and try another one
        for u, v in zip(self.halton(self.N, h1), self.halton(self.N, h2)):
            if A.sdf(u,v) < 0.0:
                if B == None or B.sdf(u,v) < 0.0:
                    # (u, v) is both in A and B (or either B is not present)
                    self.p.append([u,v])
            if len(self.p) == self.N:
                break
        self.p = math.tensor(self.p,
                            instance(i=N) & channel(vector='x,y'))

    def get_sample_points(self):
        return self.p

    # Generates the b base Halton-sequence
    # source: https://en.wikipedia.org/wiki/Halton_sequence
    def halton(self, n, b):
        n, d = 0, 1
        while True:
            x = d - n
            if x == 1:
                n = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n = (b + 1) * y - x
            yield n / d

'''
pos: lower left corner of the encompassing rectangle
size: size by which to scale encompassing rectangle
'''
class Circle:
    def __init__(self, pos=(1,1), size=1):
        self.pos = pos
        self.size = size
        self.radius = size/2

    '''
    (u, v): sample point in [0,1]x[0,1]
    return: SDF(u,v) which is
        - > 0 outside the shape
        - = 0 on the border of the shape
        - < 0 inside the shape
        with a sphere r=0.5, centered at o=[0.5, 0.5]
    '''
    def sdf(self, u, v):
        o = [0.5, 0.5]
        r = 0.5
        # distance from center of the circle
        dist = math.sqrt((u-o[0])**2 + (v-o[1])**2)
        return dist - r

    def create_points(self, p_sample):
        self.p = self.f(p_sample)
        return self.p

    def f(self, p):
        scale = tensor([self.size, self.size], channel(p))
        tx, ty = self.pos
        translate = tensor([tx, ty], channel(p))
        return (p * scale) + translate

    def get_smoke(self, domain=Box(x=math.PI, y=math.PI), x=100, y=100):
        r = self.radius
        center = tensor([self.pos[0]+r, self.pos[1]+r], channel(vector='x,y'))
        smoke = CenteredGrid(
            Sphere(center=center, radius=r),
            extrapolation.BOUNDARY,
            x=x, y=y,
            bounds=(domain)
        )

        return smoke

    def get_trivial_points(self):
        return math.tensor([
            [0.5, 0.5], # center
            [0.885, 0.885], # upper right
            [0.115, 0.885], # upper left
            [0.115, 0.115], # lower left
            [0.885, 0.115], # lower right
        ], instance(i=5) & channel(vector='x,y'))


'''
pos: (x,y) position of lower left corner
size: side length of the square
'''
class Square:
    def __init__(self, pos=(1,1), size=1):
        self.pos  = pos
        self.size = size

    '''
    (u, v): sample point in [0,1]x[0,1]
    return: SDF(u,v) which is
        - > 0 outside the shape
        - = 0 on the border of the shape
        - < 0 inside the shape
        with a sphere r=0.5, centered at o=[0.5, 0.5]
    '''
    def sdf(self, u, v):
        # All of the unit square is inside
        return -1

    def create_points(self, p_sample):
        self.p = self.f(p_sample)
        return self.p

    def f(self, p):
        scale = tensor([self.size, self.size], channel(p))
        tx, ty = self.pos
        translate = tensor([tx, ty], channel(p))
        return (p * scale) + translate

    def get_smoke(self, domain=Box(x=math.PI, y=math.PI), x=100, y=100):
        lower = tensor([self.pos[0], self.pos[1]], channel(vector='x,y'))
        upper = tensor([self.pos[0]+self.size, self.pos[1]+self.size], channel(vector='x,y'))
        smoke = CenteredGrid(
            Box(lower=lower, 
            upper=upper),
            extrapolation.BOUNDARY,
            x=x, y=y,
            bounds=(domain)
        )

        return smoke

    def get_trivial_points(self,):
        return math.tensor([
            [0.5, 0.5], # center
            [1, 1], # upper right
            [0, 1], # upper left
            [0, 0], # lower left
            [1, 0], # lower right
        ], instance(i=5) & channel(vector='x,y'))

'''
pos: lower left corner of the encompassing rectangle
'''
class Triangle():
    def __init__(self, pos=(1,1), size=1.0):
        self.pos   = pos
        self.size = size

    '''
    (u, v): sample point in [0,1]x[0,1]
    return: SDF(u,v) which is
        - > 0 outside the shape
        - = 0 on the border of the shape
        - < 0 inside the shape
        with a sphere r=0.5, centered at o=[0.5, 0.5]
    '''
    def sdf(self, u, v):
        # needs to be under both the left and right sides
        y_left  = +2*u
        y_right = -2*u + 2
        if v < y_left and v < y_right and v > 0:
            return -1 #inside triangle
        return 1 #outside triangle

    def create_points(self, p_sample):
        self.p = self.f(p_sample)
        return self.p

    def f(self, p):
        scale = tensor([self.size, self.size], channel(p))
        tx, ty = self.pos
        translate = tensor([tx, ty], channel(p))
        return (p * scale) + translate

    def get_smoke(self,domain=Box(x=math.PI, y=math.PI), x=100, y=100):
        smoke = CenteredGrid(
            self.get_marker,
            extrapolation.BOUNDARY,
            x=x, y=y,
            bounds=(domain)
        )
        return smoke

    def get_marker(self, p):
        u, v = p.vector['x'], p.vector['y']
        y_left  = +2*(u-self.pos[0]) + self.pos[1]
        y_right = -2*(u-self.pos[0]) + 2 + self.pos[1]
        bool_inside = (v < y_left) & (v < y_right) & (v > self.pos[1])
        bool_inside = math.all(bool_inside, 'vector')

        return bool_inside

    def get_trivial_points(self):
        return math.tensor([
            [0.5, 0.5], # center
            [0.5, 1], # upper right
            [0.5, 1], # upper left
            [0, 0], # lower left
            [1, 0], # lower right
        ], instance(i=5) & channel(vector='x,y'))

# TODO
def get_f_moon():
    return

# TODO BME logo
