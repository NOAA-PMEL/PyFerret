	SUBROUTINE TM_AXIS_ORDER( lines4, order, ndim, vname )

* based on 1 to 4 axes pointed to by lines4 try to determine how to place
* this object in 4 dimensional space
* e.g. if the axes are X,Y and T then slot 3 should probably be NORMAL
* suppose further that axis 2 has no units or clues to its orientation
* then our problem is to map  3 dimensions "XNT" to an ordering of 1240

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 3/92 for TMAP library
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef

#ifdef unix
        include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
        include 'tmap_format/gt_lib.parm'
        include 'tmap_format/xtm_grid.cmn_text'
        include 'tmap_format/xio.cmn_text'
        include 'tmap_format/xrisc_buff.cmn'
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XIO.CMN'
        INCLUDE 'TMAP_FORMAT:XRISC_BUFF.CMN'
#endif

        INTEGER tt  ! nice short name
        EQUIVALENCE (tt,lunit_errors)

* calling argument declarations:
      INTEGER lines4(4), order(4), ndim
      CHARACTER*(*) vname

* internal variable declarations:
      INTEGER TM_AXIS_DIRECTION, idim, lasdim, vlen

* initialize
      DO 10 idim = 1, 4
 10   order(idim) = unspecified_int4

* ... pass 1:  e.g. map to 1 0 4 -111
      DO 100 idim = 1, MIN(4, ndim)
 100  order(idim) = TM_AXIS_DIRECTION(lines4(idim))

* ..  pass 2: e.g. map "N" to "Y" to give 1 2 4 0
      lasdim = 0
      DO 200 idim = 1, 4
         IF ( order(idim).EQ.0 ) THEN
            lasdim = lasdim + 1
            order(idim) = lasdim
         ELSEIF (order(idim).GT.0 ) THEN
            lasdim = order(idim)
         ELSE
            order(idim) = 0
         ENDIF
 200  CONTINUE

* ... pass 3: if out of order (e.g. YXT given) a problem
*             ==> remap to most basic
      lasdim = 0
      DO 300 idim = 1, 4
         IF ( order(idim).EQ.0 ) GOTO 300
         IF ( lasdim .GE. order(idim) ) GOTO 500
         lasdim = order(idim)
 300  CONTINUE

* success
      RETURN

* failure - select most basic order
 500  DO 600 idim = 1, 4
         IF ( idim .LE. ndim ) THEN
            order(idim) = idim
         ELSE
            order(idim) = 0
         ENDIF
 600  CONTINUE

* sound a warning horn
      risc_buff = vname
      vlen = LEN(vname)
      CALL TM_NOTE( 'unsupported ordering of axes in variable '
     .               //risc_buff(:vlen), tt )
      CALL TM_NOTE( 'The default ordering will be used', tt )

      RETURN
      END
