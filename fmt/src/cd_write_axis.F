	SUBROUTINE CD_WRITE_AXIS ( cdfid, grid, idim, recdim,
     .                             loss, hiss, ax_type, status )


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
* Write an axis definition out to the current output netCDF file
* May be a full "parent" definition, a sub_axis, or a "child" (a sub-axis
* which points to the full axis)

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*   10/92: for performance reasons the "point_spacing" attribute will not be
*          maintained on the record axis (see cd_rec_synch)
*        - and CD_SET_MODE used to put into define mode before writing attribs
*   12/92: write axes in double precision (esp. for time axis issues) 
*    3/93: check error status after creating dimension
*    4/93: check "line_direction" in determining if time_origin is needed
*    1/95: change logical argument into integer "ax_type"
*    2/95: write NOAA COOP-compliant files: "days since 1992-02-28",
*	   "degrees_east", "degrees_north"
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* v4.91 *kob* 12/97 - added maxchildlen paramater for the max length of 
*                     a child axis
* v500 *kob*  3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,,
*                    name to 64 from 24 to match longer line_name
* V530 *sh* 12/00 - avoid performance problems of switching between netCDF
*	DEFINE and DATA mode by deferring actual coordinate writes until later


* argument definitions:
*     cdfid   - netCDF id number for already opened CDF file
*     grid    - TMAP grid id
*     idim    - axis orientation (1-4)
*     recdim  - is this to be the CDF record axis ? (input)
*     loss    - lo subscript (child axis, only)
*     hiss    - hi subscript (child axis, only)
*     status  - return status

* include files
#ifdef unix
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
	include 'xunits.cmn_text'
#else
xx        include 'netcdf.decl'  ! untested
#include "netcdf.inc"
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
        INCLUDE 'TMAP_FORMAT:CDF_TMAP.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:CD_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
#endif

* argument declarations
      LOGICAL recdim
      INTEGER cdfid, grid, idim, loss, hiss, ax_type, status

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      LOGICAL TM_ABSTRACT_AXIS, TM_DATE_OK, abstract, do_warn
      PARAMETER (do_warn = .TRUE. )
      INTEGER TM_LENSTR1, CD_AXLEN, NCDID, NCVID, NCDDEF, NCVDEF,
     .        dimid, iaxis, axid, cdfstat, lo, hi,
     .        nlen, elen, axlen, slen, edgid, npts
!      INTEGER vartype, nvdims, vdims(6), nvatts
!      LOGICAL condensed
!      REAL rlen
* *kob* 3/99 - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME,,
*                    name to 64 from 24 to match longer line_name
      CHARACTER CD_CHILDAX_NAME*64, CD_AXIS_NAME*64,
     .          name*64, units*48
!      REAL*8 val8_cdf, val8_tmap
* 12/97
      INTEGER maxchildlen
      PARAMETER (maxchildlen = 24)

* initialize
      iaxis = grid_line(idim, grid)
      name = CD_AXIS_NAME( grid, idim, nlen )
      npts = line_dim(iaxis)
      abstract = TM_ABSTRACT_AXIS( iaxis )

* determine lo/hi limits of the axis to be written
      IF ( abstract ) THEN
         lo = loss
         hi = hiss
      ELSEIF ( ax_type .EQ. pcdf_full_span ) THEN
         lo = 1
         hi = line_dim(iaxis)
      ELSE
         lo = loss
         hi = hiss
      ENDIF
!      condensed = line_regular(iaxis) .AND. (hi-lo).GT.3000

* dimension name and size for this axis
      axlen = hi - lo + 1
      IF ( abstract .OR. ax_type.NE.pcdf_full_span ) THEN
         name = CD_CHILDAX_NAME( name, lo, hi, nlen )
* *kob* 12/9/97 - use new paramater maxchildlen in length check
         nlen = MIN(nlen,maxchildlen)
      ENDIF
      dimid = NCDID( cdfid, name(:nlen), cdfstat )

* already defined ? (limited consistency checking)
      IF ( cdfstat .EQ. NCNOERR ) THEN
* ... is there a coordinate variable by the same name ?
         axid = NCVID( cdfid, name(:nlen), cdfstat )
         IF ( cdfstat .NE. NCNOERR ) GOTO 5300
         IF ( .NOT.recdim ) THEN
* ... check the length
            slen = CD_AXLEN( cdfid, dimid, status )
            IF ( status .NE. merr_ok ) GOTO 5900
            IF ( axlen .NE. slen ) GOTO 5100
!* ... check the first value on the axis for consistency
! (check fails with "Attribute not found" in CD_RD_R8_1 - not debugged *sh*)
! (also note: this may require a mode change 10/92)
!            CALL NCVINQ(cdfid, axid, name, vartype, nvdims, vdims,
!     .                 nvatts,cdfstat)
!            CALL CD_RD_R8_1( cdfid, axid,   1, vartype, name(:nlen),
!     .                        val8_cdf, status )
!            IF ( status .NE. merr_ok ) GOTO 5900
!            val8_tmap = TM_WORLD( lo, grid, idim, box_middle )
!            IF ( SNGL(val8_cdf) .NE. SNGL(val8_tmap) ) GOTO 5100
         ENDIF
         GOTO 1000
      ENDIF

* put CDF into define mode so axis can be defined
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* define the dimension
      IF ( recdim ) THEN
         dimid = NCDDEF( cdfid, name(:nlen), NCUNLIM, cdfstat )
!      ELSEIF ( condensed ) THEN
!         dimid = NCDDEF( cdfid, name(:nlen), 2, cdfstat )
      ELSE
         dimid = NCDDEF( cdfid, name(:nlen), axlen, cdfstat )
      ENDIF
      IF ( cdfstat .NE. NCNOERR ) GOTO 5200

* create the coordinate variable (12/92 -> double precision)
      axid = NCVDEF(cdfid, name(:nlen), NCDOUBLE, 1, dimid, cdfstat)
      IF ( cdfstat .NE. NCNOERR ) GOTO 5200

* assign its attributes
      IF (ax_type .EQ. pcdf_child_span) THEN
         CALL NCAPTC( cdfid, axid, 'child_axis', NCCHAR, 1, ' ',
     .        cdfstat )
      ENDIF
!      IF ( condensed ) THEN
!         CALL NCAPT( cdfid, axid, 'true_size', NCLONG, 1, npts,
!     .        cdfstat )
!      ENDIF
      IF ( line_units(iaxis) .NE. ' ' ) THEN
	 units = line_units(iaxis)	 
	 IF ( line_direction(iaxis) .EQ. 'WE'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_east'
	 ELSEIF ( line_direction(iaxis) .EQ. 'SN'
     .	.AND. line_unit_code(iaxis) .EQ. pun_degrees ) THEN
	    units = 'degrees_north'
	ELSEIF ( line_direction(iaxis) .EQ. 'TI'
     . .AND. TM_DATE_OK( line_T0(iaxis) )    ) THEN
            slen = TM_LENSTR1( line_units(iaxis) )
	    units = line_units(iaxis)(:slen)//' since '
	    CALL TM_DATE_REFORMAT(line_T0(iaxis),pdate_vax,
     .				  units(slen+8:),pdate_numslashnum,
     .				  status )
	    IF ( status .NE. merr_ok ) GOTO 5900
	ENDIF
        slen = TM_LENSTR1( units )
        CALL NCAPTC( cdfid, axid, 'units', NCCHAR, slen,
     .        	      units(:slen), cdfstat )
      ENDIF
      IF ( idim .EQ. t_dim ) THEN                  ! 4/93 update
	IF ( line_direction(iaxis) .EQ. 'TI'
     . .AND. TM_DATE_OK( line_T0(iaxis) )    ) THEN
            slen = TM_LENSTR1( line_T0(iaxis) )
            CALL NCAPTC( cdfid, axid, 'time_origin', NCCHAR, slen,
     .           line_T0(iaxis)(:slen), cdfstat )
	ENDIF
      ENDIF
      IF ( line_modulo(iaxis) ) THEN
         CALL NCAPTC( cdfid,axid,'modulo',NCCHAR,1,' ', cdfstat )
      ENDIF
      IF ( idim.EQ.z_dim .AND. line_direction(iaxis).EQ.'UD') THEN
         CALL NCAPTC( cdfid, axid, 'positive', NCCHAR, 4,
     .                'down', cdfstat )
      ENDIF

* record axis (time, usually) doesn't get cooridnate spacing or any
* coordinates inserted at creation time
      IF ( recdim ) THEN
! (code removed 10/92 - shouldn't have been here, anyway, I think *sh*)
!* ... this attribute is maintained by cd_rec_synch
!         CALL NCAPTC( cdfid, axid, 'point_spacing', NCCHAR, 4,
!     .             'even', cdfstat )
         CONTINUE
      ELSE
        IF ( line_regular(iaxis) ) THEN
           CALL NCAPTC( cdfid, axid, 'point_spacing', NCCHAR, 4,
     .             'even', cdfstat )
        ELSE
* sub-span axis may not have irregular spacing just cuz parent does
* 12/00 NO. ... regular appearing coordinates are not necessarily regular --
*       for example 2 points extracted from an irregular axis will always
*	appear to be regular, but their box limits are not necessarily so
           IF ( ax_type .EQ. pcdf_full_span
     .     .OR. ax_type .EQ. pcdf_sub_span ) THEN
              CALL NCAPTC( cdfid, axid, 'point_spacing', NCCHAR, 6,
     .             'uneven', cdfstat )
           ENDIF
        ENDIF

* set up for deferref writing of coordinate values (12/00)
	CALL CD_DEFER_COORD_WRITE(axid, grid, idim, lo, hi,
     .				  pcd_defer_points)

      ENDIF

* does it need axis box edge definitions ?
      IF (  (ax_type .EQ. pcdf_full_span
     .  .OR. ax_type .EQ. pcdf_sub_span)
     ..AND. .NOT.line_regular(iaxis)
     ..AND. .NOT.recdim ) THEN
         IF ( status .NE. merr_ok ) RETURN
         name = name(:nlen)//'edges'
         elen = nlen + 5
         dimid = NCDDEF( cdfid, name(:elen), axlen+1, cdfstat )
         edgid = NCVDEF(cdfid, name(:elen), NCDOUBLE, 1, dimid, cdfstat)
         CALL NCAPTC( cdfid, edgid, 'edges', NCCHAR, 1, ' ', cdfstat )
* ... point to it from axis coordinates definition
         CALL NCAPTC( cdfid, axid, 'edges', NCCHAR, elen,
     .                name(:elen), cdfstat )

* set up for deferref writing of edge coordinate values (12/00)
	CALL CD_DEFER_COORD_WRITE(edgid, grid, idim, lo, hi,
     .				  pcd_defer_edges)

      ENDIF

* successful completion
 1000 status = merr_ok
      RETURN

* error messages
 5100 CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_WRITE_AXIS', unspecified_int4,
     .   unspecified_int4,
     .  'axis '//name(:nlen)//' doesnt match CDF file',
     .  no_errstring, *5900 )

 5200        CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_WRITE_AXIS', cdfid,
     .       unspecified_int4, 'Failed creating coord variable '
     .       //name(:nlen), no_errstring, *5900 )

 5300 CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_WRITE_AXIS', unspecified_int4,
     .   unspecified_int4,
     .  'Bad CDF file: dimension '//name(:nlen)
     .  //' exists without coordinates',
     .  no_errstring, *5900 )

* error exit
 5900   RETURN
	END
