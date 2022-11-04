from phi.torch.flow import *

class Eigenfluid():
    '''
    N: Number of basis fields
    domain: original domain size (pi x pi)
    sampling_size: For reconstructing the velocity field
    visu_size: for visualizing the velocity field
    '''
    def __init__(self, N, DOMAIN, SAMPLING_SIZE, init_w='random'):
        # Number of basis fields
        self.N = N
        self.N_sqrt = int(math.sqrt(N))
        # Domain sizes
        self.DOMAIN = DOMAIN
        self.SAMPLING_SIZE = SAMPLING_SIZE

        # Initialize basis fields
        # N x [k1, k2, eig]
        self.basis_fields = self.get_initial_basis_fields()

        # Initialize basis coefficient vector
        if init_w == 'random':
            self.w = self.get_initial_w(random=True)
        if init_w == 'zero':
            self.w = self.get_initial_w(zero=True)

        # Precalculate advection tensor
        # TODO use sparse matrices
        self.C_k = np.zeros((N,N,N))
        self.precalculate_advection()
        self.C_k = tensor(self.C_k, instance('k'), channel('h,i'))


    # Initialize basis fields
    # First, generate the data as a regular array, 
    #  with dimensions: N x [k1,k2,eig] = N x 3
    # Assume N is a perfect square, and use all basis fields
    #  with wave number (k1,k2) up to (sqrt(N), sqrt(N))
    def get_initial_basis_fields(self):
        basis_array = []
        for k1 in range(1, self.N_sqrt+1):
            for k2 in range(1, self.N_sqrt+1):
                # append (k1, k2, eig)
                basis_array.append([k1, k2, -(k1**2 + k2**2)])
        # N instances of [k1, k2, eig] tensors
        return math.tensor(basis_array, instance(i=self.N), channel(k='k1,k2,eig'))

    # (k1, k2) -> index in basis_fields tensor
    def index_lookup(self, k1, k2):
        k = tensor([k1, k2], channel(k='k1,k2'))
        for i, f in enumerate(self.basis_fields.i):
            if all(f['k1,k2'] == k):
                return i
        return -1

    # Initialize the w basis field coefficient vecotr
    def get_initial_w(self, random=False, seed=42, zero=False):
        if zero:
            return math.zeros(instance(k=self.N))
        math.seed(seed)
        if random:
            # Scaled by 1/N
            return math.random_normal(instance(k=self.N))/self.N
            #return math.random_uniform(instance(k=N), low=-1.0, high=1.0)
        # short form:
        # w = tensor([1.0 if i == 3 or i==2 or i==5 else 0. for i in range(N)])
        w = []
        for i in range(self.N):
            if i == 2: w.append(1.3)
            elif i == 4: w.append(.7)
            elif i == 5: w.append(2.3)
            elif i == 6: w.append(.4)
            else: w.append(0)
        w = tensor(w, instance(k=self.N)) / self.N
        return w

    # p.shape should be (channel('vector'='x,y')
    #   with optional batch dimensions, e.g. (batch(i), channel(..))
    def get_phi_at(self, p):
        phi = self.phi_template(self.w, self.N, self.basis_fields)
        return phi(p)

    def get_phi(self):
        return self.phi_template(self.w, self.N, self.basis_fields)

    # The base function phi, scaled by w[i]
    def phi_template(self, w, N, basis_fields):
        def phi(p):
            vel = math.zeros_like(p)
            for i in range(N):
                k1, k2, eig = basis_fields.i[i]
                k1x, k2y = k1*p.vector['x'], k2*p.vector['y']
                # note: we could store this factor
                fact = 2/math.PI/math.sqrt(-eig)
                vel += w[i] * tensor([
                    -fact * k2 * math.sin(k1x) * math.cos(k2y),
                    +fact * k1 * math.cos(k1x) * math.sin(k2y)
                ], channel(vector='x,y'))
            return vel
        return phi

    # Return the reconstructed velocity field (SAMPLING_SIZE**2)
    def reconstruct_velocity(self, w=None, N=None):
        if w is None:
            w = self.w
        if N is None:
            N = self.N
        phi = self.phi_template(w, N, self.basis_fields)
        velocity = CenteredGrid(phi,
                                extrapolation.BOUNDARY,
                                x=self.SAMPLING_SIZE, y=self.SAMPLING_SIZE,
                                bounds=self.DOMAIN)
        return velocity

    def precalculate_advection(self):
        # Helper for calculating coeff values
        def coefdensity(h1, h2, i1, i2, c):
            if c == 0: return -0.25 * (h1*h2 - h2*i1)
            if c == 1: return +0.25 * (h1*h2 + h2*i1)
            if c == 2: return -0.25 * (h1*h2 + h2*i1)
            if c == 3: return +0.25 * (h1*h2 - h2*i1)
        for h in range(self.N):
            h1, h2, h_eig = self.basis_fields.i[h]
            for i in range(self.N):
                i1, i2, i_eig = self.basis_fields.i[i]
                # the C_k matrices corresponding to these
                # wavelengths are to be updated
                ap = [ # antipairs -- ordered as per the paper, and not the original implementation
                    [h1+i1, h2+i2],
                    [h1+i1, h2-i2],
                    [h1-i1, h2+i2],
                    [h1-i1, h2-i2],
                ]
                for c in range(4):
                    index = self.index_lookup(ap[c][0], ap[c][1])
                    # discard if wavelength is not in the span of the basis fields
                    #if ap[c][0]<N_sqrt and ap[c][1]<N_sqrt:
                    if index != -1:
                        coef = coefdensity(h1, h2, i1, i2, c)
                        eig_inv = 1 / self.basis_fields.i[i][2]
                        coef *= eig_inv
                        # TODO store in sparse matrix
                        self.C_k[index, h, i] = coef

    def step_w_euler(self, w, dt = 0.2, viscosity = .0):
        # store kinetic energy of the velocity field
        e_1 = math.l2_loss(w)*2 # the pytorch function l2_loss calculates sum(sqr(x))/2 

        # Matrix-vector product for advection
        dw = math.dot(math.dot(w, ['k'], self.C_k, ['h']), ['i'], w, ['k'])
        # Explicit Euler Integration 
        # TODO implement RK4
        w += dw * dt

        # Energy after time step
        e_2 = math.l2_loss(w)*2
        # Renormalize energy + epsilon for numerical stability
        w *= math.sqrt(e_1/e_2 + 1e-5)

        # Dissipate energy for viscosity
        if viscosity > 0:
            eig = rename_dims(self.basis_fields.k['eig'], 'i', 'k')
            w *= math.exp(eig * dt * viscosity)

        # TODO add external forces here?
        # w += f

        return w

