	INTEGER FUNCTION TM_AXIS_DIRECTION( iaxis )

* decode the "line_direction" to a simple axis

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 3/92 for TMAP library
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/xtm_grid.cmn_text'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN_TEXT'
#endif

* calling argument declarations:
      INTEGER  iaxis

* internal variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      INTEGER STR_UPCASE, idim, istat
      CHARACTER*1 c1, ww_dim_name(4) 
      DATA  ww_dim_name/'X','Y','Z','T'/
      CHARACTER*2 orient

* initialize
      orient = line_direction(iaxis)

* check the obvious
      DO 100 idim = 1, 4
         IF ( orient .EQ. axis_orients(idim) ) THEN
            TM_AXIS_DIRECTION = idim
            RETURN
         ENDIF
 100  CONTINUE

* other previously determined orientations
      IF ( orient .EQ. 'UD' ) THEN
         TM_AXIS_DIRECTION = z_dim
      ELSEIF (orient .EQ. 'EW' ) THEN
         TM_AXIS_DIRECTION = x_dim
      ELSE
         TM_AXIS_DIRECTION = 0
      ENDIF

* last ditch attempt based on axis name and units
      IF ( TM_AXIS_DIRECTION .EQ. 0 ) THEN
         istat = STR_UPCASE( c1, line_name(iaxis)(1:1) )
         DO 200 idim = 1, 4
 200     IF ( c1 .EQ. ww_dim_name(idim) ) TM_AXIS_DIRECTION = idim
      ENDIF

      RETURN
      END
