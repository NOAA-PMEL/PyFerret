C v5df.h

C Include file for using v5d functions from FORTRAN programs


C Function prototypes.  See the README file for details.  These are
C the functions you'll want to use for writing v5d file converters.

c Note: Ansley Manke 10/98. Ferret-external-functions version of this file, with 
c MAXVARS changed to MAX_V5_VARS. Make the same change in v5d.h, v5d.c, and 
c v5df_fer.h


      integer v5dcreate

      integer v5dcreatesimple

      integer v5dwrite

      integer v5dmcfile

      integer v5dclose


C 5-D grid limits, must match those in v5d.h!!!
      integer MAX_V5_VARS, MAXTIMES, MAXROWS, MAXCOLUMNS, MAXLEVELS

      parameter (MAX_V5_VARS=200)
      parameter (MAXTIMES=400)
      parameter (MAXROWS=400)
      parameter (MAXCOLUMNS=400)
      parameter (MAXLEVELS=400)

C Missing values
      real MISSING
      integer IMISSING

      parameter (MISSING=1.0E35)
      parameter (IMISSING=-987654)

