	SUBROUTINE CD_GET_TMAP_PARMS ( dset, cdfid, status )

* read special TMAP global attributes from the open netCDF file
* these attributes are:
* TMAP_expnum = "0056"
* TMAP_slice = "  SA"
* TMAP_source = "PHIL/SEIG"
* TMAP_subsource = "TROPICAL PACIFIC"
* TMAP_**anything_else**  = auxiliary parameter

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.

* argument definitions:
*       dset    - pointer to TMAP data set
*       cdfid   - returned pointer to open CDF file
*	status	- return status

* argument declarations:
	INTEGER	      dset, cdfid, status

* include files
#ifdef unix
        include 'tmap_format/netcdf.decl'
        include 'tmap_format/netcdf.inc'
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
        include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
        include 'tmap_format/xio.cmn_text'
        external xio_data
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
        INCLUDE 'TMAP_FORMAT:NETCDF.INC'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
#endif

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTRIB, got_it,
     .        do_warn
      PARAMETER (do_warn = .TRUE. )
      INTEGER nparm, iatt,
     .        ndims, nvars, ngatts, recdim, cdfstat, i
      CHARACTER*13 TM_STRING
      CHARACTER    name*64, tmap_att*64, right_side*132

      INTEGER tt                ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
        nparm = 0
        DO 10 i = 1, num_parm
           ds_parm_name(i, dset) = ' '
           ds_parm_text(i, dset) = ' '
 10     CONTINUE
	ds_source_class(dset)	= ' '
	ds_source(dset)		= ' '
	ds_subsource(dset)	= ' '
	ds_expnum(dset)		= ' '
	ds_modnum(dset)		= ' '

* get vital file statistics (cdfstat checked in last call)
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )

* loop through global attributes looking for ones beginning with TMAP
        DO 100 iatt = 1, ngatts

* get its name
           CALL NCANAM( cdfid, NCGLOBAL, iatt, name, cdfstat )
           IF ( name(:5) .NE. 'TMAP_' ) GOTO 100

* get its value
           got_it = CD_GET_ATTRIB ( cdfid, NCGLOBAL, name, do_warn,
     .          'TMAP*', right_side, 132 )

* what is it ?
           tmap_att = name(6:)
           IF ( tmap_att(:6) .EQ. 'expnum' ) THEN
              ds_expnum(dset) = right_side
           ELSEIF ( tmap_att(:5) .EQ. 'slice' ) THEN    ! "modnum" in GT sets
              ds_modnum(dset) = right_side
           ELSEIF ( tmap_att(:6) .EQ. 'source' ) THEN
              ds_source(dset) = right_side
           ELSEIF ( tmap_att(:9) .EQ. 'subsource' ) THEN
              ds_subsource(dset) = right_side
           ELSE
              nparm = nparm + 1
              IF ( nparm .GT. num_parm ) GOTO 5100
              ds_parm_name(nparm, dset) = tmap_att
              ds_parm_text(nparm, dset) = right_side
           ENDIF

 100    CONTINUE

* successful completion
 1000   status = merr_ok
        RETURN

* error exit
 5100      CALL TM_NOTE( 'maximum number of TMAP CDF attributes '
     .          //'exceeded: '//TM_STRING(DBLE(num_parm)), tt )
           GOTO 1000   ! not fatal

           END
