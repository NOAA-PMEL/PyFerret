C!ACM  Version of v5df.h with MAXVARS changed to MAX_V5_VARS.  Ferret already
C      has a parameter named MAXVARS.  The following is from Vis5D version 5.0 

C v5df.h
C Note:  Ansley Manke 10/98.  Ferret-external-functions version of the file
C v5df.h   Change MAXVARS to MAX_V5_VARS  for compatibility with ferret.
C Make the same change in v5d.h, v5d.c, and v5df_fer.h
 
C Include file for using v5d functions from FORTRAN programs


C Function prototypes.  See the README file for details.  These are
C the functions you'll want to use for writing v5d file converters.

C!ACM type declarations are made in writev5d.F to avoid compiler warnings
C     about local variable name never used.

c      integer v5dcreate
c      integer v5dcreatesimple
c      integer v5dwrite
c      integer v5dmcfile
c      integer v5dclose


C 5-D grid limits, must match those in v5d.h!!!
      integer MAX_V5_VARS, MAXTIMES, MAXROWS, MAXCOLUMNS, MAXLEVELS

      parameter (MAX_V5_VARS=30)
      parameter (MAXTIMES=400)
      parameter (MAXROWS=400)
      parameter (MAXCOLUMNS=800)
      parameter (MAXLEVELS=100)

C Missing values
      real MISSING
      integer IMISSING

      parameter (MISSING=1.0E35)
      parameter (IMISSING=-987654)

