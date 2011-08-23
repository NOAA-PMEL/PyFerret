.c:
	cc $< -g \
	/usr/local/netcdf_413/lib/libnetcdf.a \
	/usr/local/netcdf_413/lib/libnetcdff.a \
	/usr/local/hdf5_187/lib/libhdf5_hl.a \
	/usr/local/hdf5_187/lib/libhdf5.a \
	/home/users/ansley/cflib/linux/lib/libcf.a \
	/home/users/ansley/cflib/linux/lib/libcf_src.a \
	/usr/local/netcdf_413/lib/libnetcdf.a \
	-L/usr/local/lib -llapack -lcblas -lf77blas -latlas \
	-I/usr/local/netcdf_413/include \
	-I/home/users/ansley/cflib/linux/include \
	-lc -lm -lcurl -lz -o $@

