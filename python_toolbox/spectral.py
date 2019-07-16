import spharm
import _spherepack
import numpy as np

class spectral:
    def __init__(self, nx, ny):
        self.spharmt = spharm.Spharmt(nx, ny)

    def _to_complex(self, data_in):
        in_shape = data_in.shape
        ndims = len(data_in.shape)
        if ndims < 2:
            raise Exception("""Input data must be a rank 3 or above array of
                            dimensions real/imaginary, y, x""")
        if ndims == 3:
            # Convert to complex type, tranpose to fortran-like indexing and add a
            #  trailing dimension
            data_complex = np.transpose(data_in[0].data+1j*data_in[1].data)[...,np.newaxis]
            return_shape = in_shape[-2:]
        if ndims > 3:
            # Convert to complex type, reshape to 3d and transpose to fortran-like
            #  dimension order
            data_complex = np.transpose(
                (data_in[...,0,:,:].data+1j*data_in[...,1,:,:].data).\
                reshape((-1,data_in.shape[-2],data_in.shape[-1])), [2,1,0])
            return_shape = in_shape[:-3]+in_shape[-2:]
        return data_complex, return_shape

    def get_grid(self, data_in):
        status = -1
        out_shape = (self.spharmt.nlat, self.spharmt.nlon)
        ndims = len(data_in.shape)
        data_complex, return_shape = self._to_complex(data_in)
        if ndims > 3:
            out_shape = return_shape[:-2]+out_shape
        # call _spherepack to convert to regular grid
        if self.spharmt.gridtype == 'regular':
            if self.spharmt.legfunc == 'stored':
                lwork = (data_complex.shape[0]*2*data_complex.shape[1]
                            *(data_complex.shape[2]+1))
                data_grid, status = _spherepack.shses(self.spharmt.nlon,
                        data_complex.real, data_complex.imag, self.spharmt.wshses, lwork)
                if status > 0:
                    msg = 'In return from call to shses in Spharmt.spectogrd ierror =  %d' % status
                    raise ValueError(msg)
        if status == -1:
            raise Exception('Failed to call _spherepack routine')
        if ndims >3:
            data_grid = data_grid.transpose([2,0,1])
        data_grid.shape = out_shape
        return data_grid

    def get_grad(self, data_in):
        out_shape = (self.spharmt.nlat, self.spharmt.nlon)
        ndims = len(data_in.shape)
        data_complex, return_shape = self._to_complex(data_in)
        if ndims > 3:
            out_shape = return_shape[:-2]+out_shape
        n_trunc = data_complex.shape[0]-1
        sh_trunc = _spherepack.twodtooned(data_complex.real, data_complex.imag, n_trunc)
        divergence = _spherepack.lap(sh_trunc, self.spharmt.rsphere)
        du, dv = self.spharmt.getuv(np.zeros(sh_trunc.shape), divergence)
        if ndims >3:
            du = du.transpose([2,0,1])
            dv = dv.transpose([2,0,1])
        du.shape = out_shape
        dv.shape = out_shape
        return du, dv

    def get_uv(self, vorticity_in, divergence_in):
        out_shape = (self.spharmt.nlat, self.spharmt.nlon)
        ndims = len(vorticity_in.shape)
        vrt, return_shape = self._to_complex(vorticity_in)
        div, return_shape = self._to_complex(divergence_in)
        if ndims > 3:
            out_shape = return_shape[:-2]+out_shape
        vrt_oned = _spherepack.twodtooned(vrt.real,vrt.imag,vrt.shape[0]-1)
        div_oned = _spherepack.twodtooned(div.real,div.imag,div.shape[0]-1)
        u,v = self.spharmt.getuv(vrt_oned,div_oned)
        u = u.reshape(out_shape)
        v = v.reshape(out_shape)
        return u,v
