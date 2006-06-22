	SUBROUTINE CD_MAKE_VAR ( cdfid, dset, varname, type, maxstrlen,
     .                           grid, lo, hi, recdim, enhead, new, 
     .                           edges_flag, do_bounds, mode_upcase_output,
     .                           status )

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
* Define a variable for the current output netCDF file

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*               10/92: set into define mode before creating variable
*  1/25/95 - *sh* added argument enhead.  If NOT enhead then don't call
*	      cd_make_grid and do make ALL necessary axes when making the
*	      child axes
*   11/95 - *sh* use CD_AXIS_NAME rather than line_name for dynamic lines
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
*  5/97 *kob* - had to use DATA rather than PARAMATER statements for character
* 		strings, ie 'ijkl', for linux f90....
* v4.91 *kob* - increase size of CD_CHILDAX_NAME,CD_AXIS_NAME lname from
*               16 to 24
* v500 *kob*  3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME, lname,
*          cdfname, vname and gname to 64 from 24 to match longer line_name
* V533 *sh* 6/01 - added support for netCDF output string variables
*		 - check for consistency of data type on APPEND operations
* V552 *acm* 5/03   increase vname to 128 chars 
* V600 *acm* 6/05 - Attribute control. Get information from linked-list structure.
* V600 *acm* 6/06 - New MODE upcase_output controls whether axis names and variable in
*                   names are upcased cdf output.

* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     varname - name of variable
*     type    - data type for variable
*    maxstrlen- if a string variable, longest string len
*     grid    - defining grid for variable
*     lo      - low indices of variable hyperslab
*     hi      - high indices of variable hyperslab
*     recdim  - which axis (1-4 or 0) is to be the CDF record axis
*     enhead  - make enhanced heading?  (with parent grid machinery)
*     new     - have we just created a new variable ?
*     status  - return status

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
	include 'cd_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xio.cmn_text'

* argument declarations
      LOGICAL enhead, new, do_bounds, edges_flag, mode_upcase_output
      INTEGER cdfid, dset, type, maxstrlen, grid, lo(4), hi(4),
     .        recdim, status
      CHARACTER*(*) varname

* local variable declarations:
      LOGICAL TM_ABSTRACT_AXIS, CD_USE_RECDIM, 
     .        itsa_slab, need_child(4), itsa_strvar
      INTEGER TM_LENSTR1, CD_ISIT_RECDIM, CD_WRITE_STRDIM,
     .        dimid(5), iaxis, cdfstat, dims, idim, varid, vartyp,
     .        nvdims, vdims(6), nvatts, nlen, glen, vlen, ax_type,
     .        lo2(4), hi2(4), what_recdim, strdim
* *kob*  - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME, lname,
*          cdfname, vname and gname to 64 from 24
* acm* vname to 128
      CHARACTER CD_CHILDAX_NAME*128, CD_AXIS_NAME*128, lname*128,
     .		cdfname*128, vname*128, gname*64, buff*128
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      CHARACTER*1     dimnam(4)
      CHARACTER*1     ww_dim_nam(4)
      DATA dimnam/'I', 'J', 'K', 'L'/
      DATA ww_dim_nam/'X', 'Y', 'Z', 'T'/

* has the variable already been created ?
      vname = varname    ! Ultrix f77 is fussy about CHAR*(*) vars
      vlen = TM_LENSTR1( vname )
      varid = NCVID( cdfid, vname(:vlen), cdfstat )
      itsa_strvar = type .EQ. NCCHAR

* already exists: limited consistency check - size of each axis, only
      new = cdfstat .NE. NCNOERR
      IF ( .NOT.new ) THEN
         CALL NCVINQ(cdfid, varid, vname, vartyp, nvdims, vdims,
     .                 nvatts,cdfstat)
         IF ( CD_USE_RECDIM( cdfid, varid, status ) ) THEN
            recdim = t_dim
         ELSE
            recdim = 0
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 5900
	 IF (type .NE. vartyp) GOTO 5050
* ... first dimension of string variable is actually max string len
	 IF (itsa_strvar) THEN
	   maxstrlen = vdims(1)
	   DO 10 idim = 1, 5
 10	   vdims(idim) = vdims(idim+1)
	   nvdims = nvdims - 1
	 ENDIF
         dims = 0
         DO 100 idim = 1, 4
            IF ( grid_line(idim,grid) .EQ. mpsnorm
     .      .OR. lo(idim) .EQ. unspecified_int4    ) GOTO 100
            dims = dims + 1
            IF ( dims .GT. nvdims ) GOTO 5100
! this check removed to cd_write_var to enable over-writing partial hyperslabs
!            IF ( idim .NE. recdim ) THEN
!               npts = CD_AXLEN( cdfid, vdims(dims), status )
!               IF ( status .NE. merr_ok ) GOTO 5900
!               IF ( npts .NE. hi(idim)-lo(idim)+1 ) GOTO 5300
!            ENDIF
 100     CONTINUE
         IF ( dims .NE. nvdims ) GOTO 5100    ! number of dimensions match ?
         GOTO 1000
      ENDIF

* if the time axis of this variable has already been created IT will 
* determine whether this variable uses the record dimension.  The request
* from the user will be over-ruled
      what_recdim = CD_ISIT_RECDIM( cdfid, dset, grid, t_dim, status )
      IF ( what_recdim .EQ. pcdf_yes ) THEN
         recdim = t_dim
      ELSEIF ( what_recdim .EQ. pcdf_no ) THEN
         recdim = 0
      ELSE
* ... the dimension has yet to be defined - user request remains
         CONTINUE
      ENDIF

* determine if this variable needs parent/child axes
      itsa_slab = .FALSE.
      DO 200 idim = 1, 4
         need_child(idim) = .FALSE.
         IF ( idim .EQ. recdim ) GOTO 200
         iaxis = grid_line(idim,grid)
         IF ( iaxis .EQ. mpsnorm ) GOTO 200
         IF ( lo(idim) .EQ. unspecified_int4 ) GOTO 200
         IF ( TM_ABSTRACT_AXIS( iaxis ) ) GOTO 200
         need_child(idim) = lo(idim) .NE. 1
     .                 .OR. hi(idim) .NE. line_dim(iaxis)
         IF ( need_child(idim) ) THEN
	    itsa_slab = .TRUE.
	    IF ( enhead .AND.
     .	     (lo(idim).LT.1 .OR. hi(idim).GT.line_dim(iaxis))) THEN
		CALL TM_NOTE( 'Modulo '//ww_dim_nam(idim)//
     .		' axis is incompatible with enhanced netCDF heading'
     .							,lunit_errors )
		CALL TM_NOTE( 'Output file may not be readable by FERRET'
     .							,lunit_errors)
	    ENDIF
 	ENDIF
 200  CONTINUE

      IF ( enhead ) THEN
* create the parent grid needed for this variable
         CALL CD_WRITE_GRID( cdfid, dset, grid, lo, hi, recdim, itsa_slab,
     .                       gname,  edges_flag, do_bounds, 
     .                       mode_upcase_output, status )
         IF ( status .NE. merr_ok ) RETURN

* create child axes where this variable doesn't run full-span on an axis
         DO 250 idim = 1, 4
            IF ( need_child(idim) ) THEN
               CALL CD_WRITE_AXIS( cdfid, dset, grid, idim, .FALSE.,
     .                   lo(idim), hi(idim), pcdf_child_span, 
     .                   .FALSE., .FALSE., mode_upcase_output, status)
               IF ( status .NE. merr_ok ) GOTO 5900
            ENDIF
 250     CONTINUE
	 IF (itsa_strvar) strdim = CD_WRITE_STRDIM(cdfid, maxstrlen, status)
         IF ( status .NE. merr_ok ) GOTO 5900
      ELSE
* 1/95 - create the full-span or sub-span axes as appropriate
         DO 260 idim = 1, 4
	    IF ( grid_line(idim, grid) .NE. mpsnorm
     .      .AND. lo(idim) .NE. unspecified_int4     ) THEN
	      IF (need_child(idim)) THEN
	         ax_type = pcdf_sub_span
	      ELSE
	         ax_type = pcdf_full_span
	      ENDIF
              CALL CD_WRITE_AXIS( cdfid, dset, grid, idim, 
     .			       idim.EQ.recdim, lo(idim), hi(idim), 
     .			       ax_type, edges_flag, do_bounds, 
     .                         mode_upcase_output, status )
              IF ( status .NE. merr_ok ) GOTO 5900
	    ENDIF
 260     CONTINUE
	 IF (itsa_strvar) strdim = CD_WRITE_STRDIM(cdfid, maxstrlen, status)
         IF ( status .NE. merr_ok ) GOTO 5900
      ENDIF

* determine the netCDF dimension id's to use for the variable
      IF (itsa_strvar) THEN
	dimid(1) = strdim   ! 1st dim of string array is character axis
	dims = 1
      ELSE
        dims = 0
      ENDIF

* get axis name. If mode upcase_output is set, then upcase the name.

      DO 300 idim = 1, 4
         iaxis = grid_line(idim,grid)
         buff = CD_AXIS_NAME( dset, grid, idim, nlen )
         lname = buff
         IF (mode_upcase_output) CALL STR_UPCASE( lname, buff)

         IF ( iaxis .EQ. mpsnorm
     .   .OR. lo(idim) .EQ. unspecified_int4 ) GOTO 300
         dims = dims + 1
         IF ( TM_ABSTRACT_AXIS( iaxis ) ) THEN
! ... could probably use lname here, too *sh* 11/95
            cdfname = CD_CHILDAX_NAME( ww_dim_nam(idim)//'AX',
     .                                 lo(idim), hi(idim), nlen )
         ELSEIF ( idim .NE. recdim
     .  .AND. ( lo(idim) .NE. 1
     .    .OR.  hi(idim) .NE. line_dim(iaxis) ) ) THEN
            cdfname = CD_CHILDAX_NAME( lname,
     .                                 lo(idim), hi(idim), nlen )
         ELSE
            cdfname = lname
            nlen = TM_LENSTR1( cdfname )
         ENDIF
         dimid(dims) = NCDID( cdfid, cdfname(:nlen), cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5200  ! it must already exist
 300  CONTINUE

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* create the new variable
      varid = NCVDEF(cdfid, vname(:vlen), type, dims, dimid, cdfstat)
      IF ( cdfstat .NE. NCNOERR ) GOTO 5400

* if the "unlimited" record dimension is in use place no limits on that axis
      DO 400 idim = 1, 4
         IF ( lo(idim) .EQ. unspecified_int4 ) THEN
            lo2(idim) = 1
            hi2(idim) = 1
         ELSE
            lo2(idim) = lo(idim)
            hi2(idim) = hi(idim)
         ENDIF
 400  CONTINUE
      IF  (  recdim .GT. 0 ) THEN
         lo2(recdim) = 0
         hi2(recdim) = 0
      ENDIF
 
* assign attributes if this is a hyperslab of the parent grid
      IF ( itsa_slab .AND. enhead) THEN
         glen = TM_LENSTR1( gname )
         CALL NCAPTC( cdfid, varid, 'parent_grid', NCCHAR, glen,
     .                gname(:glen), cdfstat )
         CALL NCAPT( cdfid, varid, 'slab_min_index', NCLONG, 4,
     .                lo2, cdfstat )
         CALL NCAPT( cdfid, varid, 'slab_max_index', NCLONG, 4,
     .                hi2, cdfstat )
      ENDIF

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5050 CALL TM_ERRMSG
     . ( merr_badgriddef, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4,
     .   'data type of variable '//vname(:vlen)//
     .   ' is inconsist. w/ CDF file',
     .   no_errstring, *5900 )

 5100 CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4,
     .   '# of dims in variable '//vname(:vlen)//
     .   ' is inconsist. w/ CDF file',
     .   no_errstring, *5900 )

 5200 CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4,
     .  'axis '//cdfname(:nlen)//' of grid for '
     .   //vname(:vlen)//' doesnt match CDF file',
     .   no_errstring, *5900 )

! 5300 CALL TM_ERRMSG
!     . ( merr_linepredef, status, 'CD_MAKE_VAR', unspecified_int4,
!     .   unspecified_int4,
!     .   dimnam(idim)//' axis length of '//vname(:vlen)//
!     .   ' inconsist. w/ CDF file',
!     .   no_errstring, *5900 )

 5400       CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_MAKE_VAR', cdfid,
     .       unspecified_int4, 'Failed creating variable '
     .       //vname(:vlen), no_errstring, *5900 )

* error exit
 5900   RETURN
	END
