# Contributed by support at BSC-CNS, Barcelona, Spain
# for build using PPC processors and SLES 9  10/2008
# They say this is the link option that I used, the 
# default one didn't work in our system


g++ -m32 -L/lib -L/usr/X11R6/lib -o ferretdods_gui \
ccr/fermain_c.o ccr/gui_init.o \
ccr/save_arg_pointers.o special/linux_routines.o dat/*.o ../fmt/src/x*.o ../ppl/plot/ppldata.o special/ferret_dispatch.o special/xmake_date_data.o special/fakes3.o special/ferret_query_f.o  special/xrevision_type_data.o special/xplatform_type_data.o \
         \
         \
        /gpfs/apps/FERRET/SRC/6.13_beta/lib/libgui.a ../list-2.1/liblist.a \
         \
        ../ppl/tmapadds/*.o  \
        ef_utility/*.o \
         \
        /gpfs/apps/FERRET/SRC/6.13_beta/lib/libxeq.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libgnl.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libferplt.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/librpn.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libstk.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libdoo.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libocn.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libctx.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libfmt.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libino.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libmem.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libutl.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libdat.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libccr.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libefi.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libefn.a \
        /gpfs/apps/FERRET/SRC/6.13_beta/lib/libplt.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libpll.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libsym.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libcmp.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libour.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libepi.a /gpfs/apps/FERRET/SRC/6.13_beta/lib/libusr.a  \
         \
        /gpfs/apps/FERRET/SRC/6.13_beta/lib/tmap_lib.a \
        ../readline-4.1/libreadline.a  \
        /usr/lib/libncurses.a \
        -L/gpfs/apps/DAP/3.8.2/32/lib /gpfs/apps/NCDAP/3.7.2/32/lib/libnc-dap.a /gpfs/apps/DAP/3.8.2/32/lib/libdap.a /gpfs/apps/DAP/3.8.2/32/lib/libdapclient.a /gpfs/apps/NCDAP/3.7.2/32/lib/libnc-dap.a /gpfs/apps/DAP/3.8.2/32/lib/libdap.a /usr/lib/libxml2.a -L/gpfs/apps/DAP/3.8.2/32/lib -L/gpfs/apps/DAP/3.8.2/32/lib -lz -lpthread /gpfs/apps/CURL/7.19.0/32/lib/libcurl.a -L/usr/lib -lssl -lcrypto -ldl -lssl -lcrypto /usr/lib/libgssapi.so.1 -lcrypto -lresolv -ldl -lz -lz  -lpthread -lz /gpfs/apps/FERRET/SRC/netcdf-3.6.2/install/lib/libnetcdf.a -L/opt/ibmcmp/xlf/10.1/lib/ -lxl \
        -L/gpfs/apps/HDF4/32/lib/  -ldf -ljpeg /gpfs/apps/HDF4/32/lib/libz.a  \
        ../xgks/src/lib/libxgks.a \
        -L/usr/X11R6/lib -ldl /usr/X11R6/lib/libXpm.a  -lc /usr/X11R6/lib/libXm.a -lXt -lXext -lX11 -lXp /usr/X11R6/lib/libXmu.a -lSM       -lICE        -lc -lg2c -lstdc++ -L/opt/ibmcmp/xlf/10.1/lib/ -lxl -lrt -Wl,-relax

