	SUBROUTINE CD_SCAN_VARS ( dset, cdfid, status )

*
*
*  This software was developed by the Thermal Modeling and Analysis
*  Project(TMAP) of the National Oceanographic and Atmospheric
*  Administration's (NOAA) Pacific Marine Environmental Lab(PMEL),
*  hereafter referred to as NOAA/PMEL/TMAP.
*
*  Access and use of this software shall impose the following
*  obligations and understandings on the user. The user is granted the
*  right, without any fee or cost, to use, copy, modify, alter, enhance
*  and distribute this software, and any derivative works thereof, and
*  its supporting documentation for any purpose whatsoever, provided
*  that this entire notice appears in all copies of the software,
*  derivative works and supporting documentation.  Further, the user
*  agrees to credit NOAA/PMEL/TMAP in any publications that result from
*  the use of this software or in any product that includes this
*  software. The names TMAP, NOAA and/or PMEL, however, may not be used
*  in any advertising or publicity to endorse or promote any products
*  or commercial entity unless specific written permission is obtained
*  from NOAA/PMEL/TMAP. The user also understands that NOAA/PMEL/TMAP
*  is not obligated to provide the user with any support, consulting,
*  training or assistance of any kind with regard to the use, operation
*  and performance of this software nor to provide the user with any
*  updates, revisions, new versions or "bug fixes".
*
*  THIS SOFTWARE IS PROVIDED BY NOAA/PMEL/TMAP "AS IS" AND ANY EXPRESS
*  OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
*  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
*  ARE DISCLAIMED. IN NO EVENT SHALL NOAA/PMEL/TMAP BE LIABLE FOR ANY SPECIAL,
*  INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER
*  RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF
*  CONTRACT, NEGLIGENCE OR OTHER TORTUOUS ACTION, ARISING OUT OF OR IN
*  CONNECTION WITH THE ACCESS, USE OR PERFORMANCE OF THIS SOFTWARE. 
*
*
* Loads common blocks XDSET_INFO and XSTEP_FILES with vital system and
* background information from a netCDF file
* It processes FERRET special attributes without regard to whether variables
* are "generic" or "FERRET"

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92 - adds a new data set format to TMAP libraries
*              - based loosely on EZ_INIT_DSET
*      4/18/92 - replaced "elif" syntax with
*                   else
*                     if
*                for sgi port   *kob*
*      8/18/92 - variable name with attribute error messages
*       3/4/93 - read global bad data flag (for EPIC files)
*              - don't process EPIC variable 'time2'
*	  5/94 - added support for DOUBLE variables
*	 11/94 - ignore the "history" attribute.  Use "long_name_mod" instead.
*	  2/95 - check data type of missing_value.  Possibly convert
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* 7/98 *sh* - allow broader flexibility in variable names through retaining
*             original case when appropriate
* 1/99 *sh* - fixed bug introduced in 7/98 changes -- blows up when file has
*		only coord variables (e.g. climatological_axes)
* 2/99 *kob* - adding call to SWITCH_NAN for NaN recognition

* argument definitions:
*       dset    - pointer to TMAP data set
*	cdfid   - id of currently open CDF file
*	status	- return status

* argument declarations:
	INTEGER	      dset, cdfid, status

* include files
#ifdef unix
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
#include "gt_lib.parm"
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
        include 'xdset_info.cd_equiv'
        include 'xio.cmn_text'
        external xio_data
#else
        INCLUDE 'TMAP_FORMAT:NETCDF.DECL'
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
        INCLUDE 'TMAP_FORMAT:XIO.CMN_TEXT'
        EXTERNAL XIO_DATA
#endif

* local variable declarations:
*  ivtm - TMAP library variable ID
*  ivcd - netCDF file variable ID
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL CD_GET_ATTRIB, CD_GET_ATTVAL, CD_GET_BOUNDS,
     .        TM_LEGAL_NAME,
     .        got_it, got_bad, got_missing, tf2, its_epic, 
     .        do_warn, got_a_variable
      PARAMETER (do_warn = .TRUE. )
      INTEGER STR_UPCASE, STR_CASE_BLIND_COMPARE, TM_LENSTR1, NCDID,
     .        ndims, nvars, cdfstat, las_pos, first_pos, idim,
     .        ivcd, ivtm, vartyp, nvdim, vdims(6), nvatts, i, vlen,
     .        ngatts, recdim, istat, npts, lasdim, attype
      CHARACTER    vname*24, buff*64
      CHARACTER*13 TM_STRING
      REAL*4  global_bad

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* special equivalence (cannot include in tmap_format/xdset_info.cd_equiv because "BYTE"
* isn't supported on DECstation
* reorder loop logic because linux cpp didn't like it *kob* 10/96

#ifdef unix
#ifdef sun
      BYTE cd_data_type(maxvars)
#else
      INTEGER*1 cd_data_type(maxvars)
#endif
#else
      BYTE cd_data_type(maxvars)
#endif
      EQUIVALENCE ( ds_precision, cd_data_type )
      REAL    r4_grid_start(4,maxvars),
     .        r4_grid_end(4,maxvars)
      EQUIVALENCE ( r4_grid_end,   ds_grid_end ),
     .            ( r4_grid_start, ds_grid_start )

* get the vital statistics for the CDF file
        CALL NCINQ( cdfid, ndims, nvars, ngatts, recdim, cdfstat )
        IF ( cdfstat .NE. NCNOERR ) GOTO 5200
        its_epic = ds_type(dset) .EQ. 'ECDF'

* get the global bad data flag (used by EPIC but sometimes missing)
        got_bad = CD_GET_ATTVAL( cdfid, NCGLOBAL, 'missing_value',
     .             do_warn, ' ', global_bad, 1, istat )
        IF ( .NOT.got_bad ) THEN
           IF ( its_epic ) THEN
              global_bad = 1.E35     ! default for EPIC
           ELSE
              global_bad = -1.E34    ! default for FERRET
           ENDIF
        ENDIF

* loop through all variables in the net CDF file
	las_pos  = 1    ! reduces redundant search for space
	got_a_variable = .FALSE.
	DO 200 ivcd = 1,nvars

* get the vital statistics for this variable
           CALL NCVINQ(cdfid, ivcd, vname, vartyp, nvdim, vdims,
     .                 nvatts,cdfstat)
           IF ( cdfstat .NE. NCNOERR ) GOTO 5200
           vlen = TM_LENSTR1(vname)

* is it a "real" variable or an CDF axis variable?
           i = NCDID( cdfid, vname(:vlen), cdfstat )
           IF (cdfstat .EQ. NCNOERR ) GOTO 200

* is it a real variable or a parent "grid" definition ?
          CALL NCDINQ( cdfid, vdims, buff, npts, cdfstat )
          istat = STR_CASE_BLIND_COMPARE('grid_definition', buff(:15) )
          IF ( istat .EQ. str_eq ) GOTO 200

* or is it really an edge definition ?
           got_it = CD_GET_ATTRIB( cdfid, ivcd, 'edges',
     .                            do_warn, vname(:vlen), buff, 64 ) !linenamlen
           IF ( got_it .AND. buff.EQ.' ' ) GOTO 200

* or is it milliseconds for an an EPIC time axis
           IF ( its_epic .AND. vname(:vlen).EQ.'time2' ) GOTO 200

* find the next storage position in TMAP common
	  DO 150 ivtm = las_pos,maxvars
	    IF (ds_var_setnum(ivtm) .EQ. set_not_open) GOTO 151
  150     CONTINUE
* oops - no room left for new variables
	  GOTO 5100

 151      las_pos = ivtm
	  IF (.NOT.got_a_variable) first_pos = ivtm
	  got_a_variable = .TRUE.
* set obvious defaults for this variable
          cd_varid(ivtm)        = ivcd ! netCDF variable ID
          ds_var_setnum(ivtm  ) = dset
          ds_var_title(ivtm)    = ' '
          ds_var_titl_mod(ivtm) = ' '
          ds_var_units(ivtm)    = ' '
	  ds_missing_flag(ivtm)	= global_bad
	  ds_bad_flag(ivtm)	= ds_missing_flag(ivtm)
	  ds_precision(ivtm)	= 'SINGLE'
          cd_scale_factor(ivtm) = 1.0
          cd_add_offset(ivtm)   = 0.0
          cd_scaled(ivtm)       = .FALSE.
	  DO 160 idim = 1,4
             ds_grid_start(idim,ivtm)= unspecified_int4
             ds_grid_end  (idim,ivtm)= unspecified_int4
  	     ds_ordering(idim,ivtm)= unspecified_int4
 160      CONTINUE

* is it a supported data type ?
          cd_data_type(ivtm) = vartyp
! DOUBLE supported as of 5/2/94
!          IF ( vartyp .EQ. NCDOUBLE ) THEN
!              CALL TM_NOTE('Unsupported netCDF data type for variable '
!     .             //vname(:vlen), tt )
!          ENDIF

* is it a supported dimensionality ?
          IF ( nvdim .GT. 4 ) THEN
             CALL TM_NOTE('Unsupported netCDF # of axes for variable '
     .             //vname(:vlen), tt )
             GOTO 200
          ENDIF

* see what generic attributes can be read: long_name, units, missing_value
          got_it = CD_GET_ATTRIB( cdfid, ivcd, 'long_name',
     .                      .NOT.do_warn, ' ', ds_var_title(ivtm), 40 )
          got_it = CD_GET_ATTRIB( cdfid, ivcd, 'long_name_mod',
     .                  .NOT.do_warn, ' ', ds_var_titl_mod(ivtm), 40 )
          got_it = CD_GET_ATTRIB( cdfid, ivcd, 'units',
     .                 do_warn, vname(:vlen), ds_var_units(ivtm), 24 )
          got_it = CD_GET_ATTVAL( cdfid, ivcd, 'scale_factor',
     .             do_warn,vname(:vlen),cd_scale_factor(ivtm),1,istat )
          tf2    = CD_GET_ATTVAL( cdfid, ivcd, 'add_offset',
     .             do_warn,vname(:vlen),cd_add_offset(ivtm),1,istat )
          cd_scaled(ivtm) = got_it .OR. tf2

* netCDF files may use "standard" attributes missing_value and/or
* _FillValue to flag bad data
* For the NOAA COOP standard the missing_value flag must be stored in the
* packed data type.  If so it must be unpacked.  (2/95)
          got_bad = CD_GET_ATTVAL( cdfid, ivcd, 'missing_value',
     .             do_warn, vname(:vlen), ds_bad_flag(ivtm), 1, istat )
          got_missing = CD_GET_ATTVAL( cdfid, ivcd, '_FillValue',
     .             do_warn, vname(:vlen), ds_missing_flag(ivtm),1,istat)
* ... need to unpack the missing_value flag?
*     (note that the netCDF library insists on _FillValue's data type)
          IF ( vartyp.LE.NCLONG .AND. cd_scaled(ivtm) ) THEN
             IF ( got_bad ) THEN
	        CALL NCAINQ(cdfid, ivcd, 'missing_value', attype,
     .			    i, istat )
	        IF ( attype .EQ. vartyp ) ds_bad_flag(ivtm) =
     .			ds_bad_flag(ivtm)*cd_scale_factor(ivtm)
     .			+ cd_add_offset(ivtm)
	     ENDIF
          ENDIF
          IF ( got_bad .AND. .NOT.got_missing )
     .        ds_missing_flag(ivtm) =  ds_bad_flag(ivtm)
          IF ( got_missing .AND. .NOT.got_bad )
     .        ds_bad_flag(ivtm) =  ds_missing_flag(ivtm)

*  *kob* call SWITCH_NAN to make sure if NaN exisits, its in ds_missing_flag
	  CALL SWITCH_NAN(ds_bad_flag(ivtm), ds_missing_flag(ivtm))

* see what special FERRET attributes can be read: axis ordering and
* sub-hyperslab bounds
          got_it = CD_GET_ATTRIB( cdfid, ivcd, 'axis_order',
     .             do_warn,vname(:vlen),buff,4 )
          lasdim = 0
          IF (got_it) THEN
             DO 170 idim = 1, 4
               IF ( buff(idim:idim) .EQ. ' ' ) THEN
                  ds_ordering(idim,ivtm) = 0
                  GOTO 170
               ELSEIF ( buff(idim:idim) .EQ. 'T' ) THEN
                  ds_ordering(idim,ivtm) = 4
               ELSE
                  ds_ordering(idim,ivtm) = ICHAR(buff(idim:idim))-87 ! XYZ->123
               ENDIF
               IF ( ds_ordering(idim,ivtm) .LT. 0
     .         .OR. ds_ordering(idim,ivtm) .GT. 4 ) THEN 
                  CALL TM_NOTE(
     .                  'Illegal netCDF "axis_order" for variable '
     .                  //vname(:vlen), tt )
                  GOTO 200
                ENDIF
                IF ( ds_ordering(idim,ivtm) .LE. lasdim ) THEN
                   CALL TM_NOTE(
     .                     '"axis_order" permutations not supported: '
     .                     //vname(:vlen), tt )
                   GOTO 200
                ENDIF
                lasdim = ds_ordering(idim,ivtm)
 170         CONTINUE
          ENDIF

* does this variable have a FERRET "parent grid" ?
* if so check for explicit hyperslab limits
          buff = ' '   ! problems from characters beyond 16
          got_it = CD_GET_ATTRIB( cdfid, ivcd, 'parent_grid',
     .                           do_warn, vname(:vlen), buff, 16 )!gridnamlen
          IF ( got_it ) THEN
             got_it = CD_GET_BOUNDS(cdfid, ivcd, ds_grid_start(1,ivtm),
     .                           ds_grid_end(1,ivtm), recdim, status )
             IF ( status .NE. merr_ok ) GOTO 5900
          ENDIF

* officially claim this variable slot
* 7/98 - save in case-sensitive form at this point -- will check over later
!        istat = STR_UPCASE( ds_var_code(ivtm), vname(:vlen) )
        ds_var_code(ivtm) = vname

 200    CONTINUE

* pass through the variable names making sure that case-sensitive
* duplicates (e.g. 'var' and 'VAR') retain case and that names which
* would be illegal variable names to Ferret are retained in original form
	IF (got_a_variable) THEN
	  DO 310 ivtm = first_pos,las_pos
	    IF (ds_var_setnum(ivtm) .NE. dset) GOTO 310
	    IF (.NOT.TM_LEGAL_NAME(ds_var_code(ivtm))) GOTO 310  ! leave as-is
* ... see if this name case-blind duplicates any other in the data set
	    DO 300 i = first_pos,las_pos
	      IF ( i .EQ. ivtm ) GOTO 300
	      istat = STR_CASE_BLIND_COMPARE(ds_var_code(ivtm),
     .                                       ds_var_code(i)    )
	      IF ( istat .EQ. str_eq ) GOTO 310                  ! leave as-is
 300	    CONTINUE
* ... (finally) upcase the name
	  istat = STR_UPCASE( ds_var_code(ivtm), ds_var_code(ivtm) )
 310    CONTINUE
	ENDIF

* successful completion
	status = merr_ok
	RETURN

* ERROR MESSAGES
 5100	CALL TM_ERRMSG ( merr_varlim, status, 'CD_SCAN_VARS',
     .			 dset, no_stepfile,
     .			 'MAX='//TM_STRING(DBLE(maxvars)),
     .			 no_errstring, *5900)

 5200   CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_SCAN_VARS', cdfid,
     .        no_varid, no_errstring, no_errstring, *5900 )

* error exit
 5900   RETURN
	END
