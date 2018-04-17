#! /bin/sh
# Copies the Homebrew .dylibs required to run the ferret executable to a
# dylibs subdirectory (creating it if not present) of the current directory.

if [ ! -d dylibs ]; then
    if [ -e dylibs ]; then
        echo "dylibs exists but is not a directory"
        exit 1
    fi
    mkdir dylibs
fi
cd dylibs

brewprefix=`brew config | awk '/HOMEBREW_PREFIX/ {print $2}'`

libdir="${brewprefix}/lib"
echo "Copying dylib libraries from ${libdir}"
for name in cairo fontconfig freetype fribidi glib-2.0 gobject-2.0 graphite2 \
            gthread-2.0 harfbuzz hdf5 hdf5_hl netcdf netcdff pango-1.0 \
            pangocairo-1.0 pangoft2-1.0 pcre pixman-1 png16 sz ; do
    echo "    ${name}"
    cp -f ${libdir}/lib${name}.*.dylib .
done

libdir="${brewprefix}/Cellar/libffi/*/lib"
echo "Copying ffi library from ${libdir}"
cp -f ${libdir}/libffi.*.dylib .

libdir="${brewprefix}/Cellar/gettext/*/lib"
echo "Copying intl library from ${libdir}"
cp -f ${libdir}/libintl.*.dylib .

libdir=`echo ${brewprefix}/opt/gcc/lib/gcc/*`
echo "Copying dylib libraries from ${libdir}"
for name in gfortran quadmath gcc_s ; do
    echo "    ${name}"
    cp -f ${libdir}/lib${name}.*.dylib .
done

