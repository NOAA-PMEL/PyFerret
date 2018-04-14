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

libdir=`echo ${brewprefix}/opt/gcc/lib/gcc/*`
echo "Copying dylib libraries from ${libdir}"
for name in quadmath gcc_s ; do
    echo "    ${name}"
    cp -f ${libdir}/lib${name}.*.dylib .
done
