	SUBROUTINE CD_MAKE_VAR ( cdfid, dset, varname, type, maxstrlen,
     .                           grid, lo, hi, recdim, enhead, new, 
     .                           edges_flag, do_bounds, mode_upcase_output,
     .                           keepax_flag, date_fmt_out,
     .                           netcdf4_type, xchunk_size, 
     .                           ychunk_size, zchunk_size, tchunk_size, 
     .                           deflate_lev, shuffle_flag, endian_code,
     .                           stringdimname, do_coords, status )

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
* V601 *acm* 11/06 - Fix bug 1471; new routine CD_AXIS_OUTNAME gets correct 
*                    output axis name.
* V602 *acm*12/06 - Bug 1478: cd_isit_recdim needs mode_upcase_output.
* V62  *acm*  2/09 - Fix bug 1634: allow SAVE/KEEP_AXISNAMES so that child axis names
*                    arent written. They can get too long for OPeNDAP library calls
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V66  *acm* 3/10  Settings for NetCDF-4 files. 
* V66  *acm* 3/10  Check: If chunk size given is larger than grid size in that
*                  direction, then its an error.
* V67  *acm* 3/11  Fix bug 1750. Revert to older behavior: if grid uses same axis
*                  name but different contents, write both to file with the name
*                  Ferret uses internally.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V685 6/13 *acm* Dont add SAVE/NOCOORDS
* V686 12/13*acm* For string variables from datasets, keep the string-dimension name and length
*                 from the original dataset on output.
* V687 3/14 *acm* Fix ticket 1750. Write Ferret-internal axis names to cdf files, unless
*                 the user has said /KEEP_AXISNAMES
* V69+ 4/14 *acm* After the above fix, need a tweak to be able to append to files having 
*                 dimensions that are not coordinate variables
* V697 2/16 *acm* Remove unused variables from the call to CD_SET_COMPRESS
* V698  4/16 *acm* Fix ticket 2241: If file has dimension but not a coordinate 
*                  variable, do not write the coordinate variable, ONLY IF writing 
*                  the full-span in that direction.
* V710  9/16 *acm* Ticket 2459: If axis is already in the file with a different
*                  direction write axis with a unique name for this direction.
* v745 *acm* 12/18 Issue 1909, option to write dates in ISO8601 format
* v751 *acm*  8/19 Issue 1946, allow for user-set string dimension length, SAVE/STRLEN
* V751 *acm* 11/19 *acm* new qualifier SAVE/NOCOORDS
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

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
*     netcdf4_type	- netcdf format: 4=netcdf4, 3=netcdf3(=classic), 6=64-bit
*     xchunk_size     - chunk size in X direction
*     ychunk_size     - chunk size in Y direction
*     zchunk_size     - chunk size in Z direction
*     tchunk_size     - chunk size in T direction
*     deflate_lev     - deflate_lev level (0 for no deflate_lev)
*     shuffle_flag    - shuffle - 0 or 1
*     endian_code     - native=0, little=1, big=0 match nf_endian_* params
*     status  - return status

* include files
        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
        include 'cdf_tmap.parm'
#include "gt_lib.parm"
	include 'cd_lib.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
		include 'xio.cmn_text'

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* argument declarations
      LOGICAL enhead, new, edges_flag, mode_upcase_output, do_coords
      INTEGER cdfid, dset, type, maxstrlen, grid, lo(nferdims), hi(nferdims),
     .        recdim, keepax_flag, do_bounds, netcdf4_type, xchunk_size, 
     .        ychunk_size, zchunk_size, tchunk_size, 
     .        deflate_lev, shuffle_flag, endian_code, date_fmt_out, status
      CHARACTER*(*) varname, stringdimname

* local variable declarations:
      LOGICAL TM_ABSTRACT_AXIS, CD_USE_RECDIM, 
     .        itsa_slab, need_child(nferdims), itsa_strvar, 
     .        name_from_grid(nferdims), dim_only(nferdims)
      INTEGER TM_LENSTR1, CD_ISIT_RECDIM, CD_WRITE_STRDIM, STR_SAME,
     .        dimid(8), iaxis, cdfstat, dims, idim, varid, vartyp,
     .        nvdims, vdims(8), nvatts, nlen, glen, vlen, ax_type,
     .        lo2(nferdims), hi2(nferdims), what_recdim, strdim, dimsize,
     .        chunk_set, duplines, jaxis, ii, slen, strdimlen
      REAL    val
* *kob*  - change lengths of CD_CHILDAX_NAME, CD_AXIS_NAME, lname,
*          cdfname, vname and gname to 64 from 24
* acm* vname to 128
      CHARACTER CD_CHILDAX_NAME*128, CD_AXIS_OUTNAME*128, CD_AXIS_NAME*128, 
     .		lname*128, cdfname*128, vname*128, gname*64, buff*128, 
     .		DIMDIR*1, TM_FMT*48, c_str*12, d_str*12

      INTEGER dlen, clen
      INTEGER chunks(nferdims)  

* *kob* explicit definitions not allowed on decl line in linux...
*       moved it to parameter line 10/96
      CHARACTER*1     dimnam(nferdims)
      CHARACTER*1     ww_dim_nam(nferdims)
      DATA dimnam/'I', 'J', 'K', 'L', 'M', 'N'/
      DATA ww_dim_nam/'X', 'Y', 'Z', 'T', 'E', 'F'/

* has the variable already been created ?
      vname = varname    ! Ultrix f77 is fussy about CHAR*(*) vars
      vlen = TM_LENSTR1( vname )
      cdfstat = NF_INQ_VARID(cdfid, vname(:vlen), varid)

      itsa_strvar = type .EQ. NF_CHAR

      new = cdfstat .NE. NF_NOERR
      
      DO idim = 1, nferdims
         dim_only(idim) = .FALSE.
         IF (.NOT.do_coords) dim_only(idim) = .TRUE.
      ENDDO

* already exists: limited consistency check - size of each axis, only
      IF ( .NOT.new ) THEN
         cdfstat = NF_INQ_VAR (cdfid, varid, vname, vartyp, 
     .                 nvdims, vdims, nvatts)
         IF ( CD_USE_RECDIM( cdfid, varid, status ) ) THEN
            recdim = t_dim
         ELSE
            recdim = 0
         ENDIF
         IF ( status .NE. merr_ok ) GOTO 5900
	 IF (type .NE. vartyp) GOTO 5050
* ... first dimension of string variable is actually max string len
	 IF (itsa_strvar) THEN
	   strdim = vdims(1)
	   cdfstat = NF_INQ_DIMLEN(cdfid, strdim, slen)
	   IF (maxstrlen .GT. slen) GOTO 5300

	   DO idim = 1, 5
 	      vdims(idim) = vdims(idim+1)
	   ENDDO
	   nvdims = nvdims - 1
	 ENDIF
         dims = 0
         DO 100 idim = 1, nferdims
            IF ( grid_line(idim,grid) .EQ. mpsnorm
     .      .OR. lo(idim) .EQ. unspecified_int4    ) GOTO 100
            dims = dims + 1
            IF ( dims .GT. nvdims ) GOTO 5100
 100     CONTINUE
         IF ( dims .NE. nvdims ) GOTO 5100    ! number of dimensions match ?
         GOTO 1000
      ENDIF

* if the time axis of this variable has already been created IT will 
* determine whether this variable uses the record dimension.  The request
* from the user will be over-ruled
      what_recdim = CD_ISIT_RECDIM( cdfid, dset, grid, t_dim, 
     .               mode_upcase_output, keepax_flag, status )
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
      DO 200 idim = 1, nferdims
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
     .                       mode_upcase_output, keepax_flag, 
     .                       date_fmt_out, status )

         IF ( status .NE. merr_ok ) RETURN

* create child axes where this variable doesn't run full-span on an axis
         DO 250 idim = 1, nferdims
            IF ( need_child(idim) ) THEN
               CALL CD_WRITE_AXIS( cdfid, dset, grid, idim, .FALSE.,
     .                   lo(idim), hi(idim), pcdf_child_span, 
     .                   .FALSE., .FALSE., mode_upcase_output, 
     .                   keepax_flag, name_from_grid(idim), dim_only(idim), 
     .                   date_fmt_out, status)
               IF ( status .NE. merr_ok ) GOTO 5900
            ENDIF
 250     CONTINUE
	 IF (itsa_strvar) strdim = CD_WRITE_STRDIM(cdfid, maxstrlen, stringdimname, status)
         IF ( status .NE. merr_ok ) GOTO 5900
      ELSE
* 1/95 - create the full-span or sub-span axes as appropriate
         DO 260 idim = 1, nferdims
	    dim_only(idim) = .FALSE.
	    IF (.NOT.do_coords) dim_only(idim) = .TRUE. 
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
     .                         mode_upcase_output, keepax_flag, 
     .                         name_from_grid(idim), dim_only(idim), 
     .                         date_fmt_out, status )
              IF ( status .NE. merr_ok ) GOTO 5900
	    ENDIF
 260     CONTINUE
	 IF (itsa_strvar) strdim = CD_WRITE_STRDIM(cdfid, maxstrlen, stringdimname, status)
         IF ( status .NE. merr_ok ) GOTO 5900
      ENDIF

      do idim = 1, nferdims+1
	 dimid(idim) = 0
      enddo

* determine the netCDF dimension id's to use for the variable
      IF (itsa_strvar) THEN
	dimid(1) = strdim   ! 1st dim of string array is character axis
	dims = 1
      ELSE
        dims = 0
      ENDIF 

* Are there grid lines duplicated in the grid (see tiket 2459 for an example)
      duplines = 0
      DO 280 idim = 1, nferdims
         iaxis = grid_line(idim,grid)
	 IF (iaxis.NE.int4_init .AND. iaxis.GT.0) THEN
	    DO ii = 1, nferdims 
	       jaxis = grid_line(ii,grid)
	       IF (ii.NE.idim .AND. jaxis.EQ.iaxis) duplines = duplines+1
	    ENDDO
	 ENDIF
  280 CONTINUE

* get axis name. If mode upcase_output is set, then upcase the name

      DO 300 idim = 1, nferdims
         iaxis = grid_line(idim,grid)
         buff = CD_AXIS_NAME( dset, grid, idim, nlen )
         IF (keepax_flag .EQ. 1) buff = CD_AXIS_OUTNAME( dset, grid, idim, nlen )

	 IF (name_from_grid(idim)) THEN 
* using the basic name straight from the grid
            buff = line_name(iaxis)
	 ENDIF
         lname = buff
         IF (mode_upcase_output) THEN
	    CALL STR_UPCASE( lname, buff)
	 ELSE
	    IF (STR_SAME(CD_AXIS_NAME(dset, grid, idim, nlen), 
     .                   CD_AXIS_OUTNAME(dset, grid, idim, nlen)) .EQ. 0) THEN
	       buff = CD_AXIS_OUTNAME(dset, grid, idim, nlen)
               lname = buff
            ENDIF
         ENDIF

	 CALL CD_AXNAME_TO_WRITE (cdfid, idim, lname, nlen, cdfstat)
	 
c	 IF (dim_only(idim)) THEN 
c	    buff = CD_AXIS_OUTNAME(dset, grid, idim, nlen)
c            lname = buff
c         ENDIF

         IF ( iaxis .EQ. mpsnorm
     .   .OR. lo(idim) .EQ. unspecified_int4 ) GOTO 300
         dims = dims + 1
         IF ( TM_ABSTRACT_AXIS( iaxis ) ) THEN
! ... could probably use lname here, too *sh* 11/95
            cdfname = lname
            IF (keepax_flag .EQ. 0) 
     .       cdfname = CD_CHILDAX_NAME( lname,
     .                                 lo(idim), hi(idim), nlen )
         ELSEIF ( idim .NE. recdim
     .  .AND. ( lo(idim) .NE. 1
     .    .OR.  hi(idim) .NE. line_dim(iaxis) ) ) THEN
            cdfname = lname
            IF (keepax_flag .EQ. 0) 
     .       cdfname = CD_CHILDAX_NAME( lname,
     .                                 lo(idim), hi(idim), nlen )
         ELSE
            cdfname = lname
         ENDIF
         nlen = TM_LENSTR1( cdfname )
         cdfstat = NF_INQ_DIMID ( cdfid, cdfname(:nlen), dimid(dims) )
         IF ( cdfstat .NE. NF_NOERR ) GOTO 5200  ! it must already exist

         chunk_set = 0
         IF (netcdf4_type .EQ. 4) THEN
            IF (line_direction(iaxis) .EQ. 'WE' .OR.  
     .              line_direction(iaxis) .EQ. 'XX') THEN
               chunks(dims) = xchunk_size
	       DIMDIR = 'X'
	       dimsize = hi(idim) - lo(idim) + 1
	       IF (chunks(dims) .GT. dimsize ) GOTO 5700
	       IF (xchunk_size .GT. 0) chunk_set = 1
            ELSEIF (line_direction(iaxis) .EQ. 'SN' .OR.  
     .              line_direction(iaxis) .EQ. 'YY') THEN
               chunks(dims) = ychunk_size
	       DIMDIR = 'Y'
	       dimsize = hi(idim) - lo(idim) + 1
	       IF (chunks(dims) .GT. dimsize ) GOTO 5700
	       IF (ychunk_size .GT. 0) chunk_set = 1
            ELSEIF (line_direction(iaxis) .EQ. 'DU' .OR.  
     .              line_direction(iaxis) .EQ. 'UD') THEN
               chunks(dims) = zchunk_size
	       DIMDIR = 'Z'
	       dimsize = hi(idim) - lo(idim) + 1
	       IF (chunks(dims) .GT. dimsize ) GOTO 5700
	       IF (zchunk_size .GT. 0) chunk_set = 1
            ELSEIF (line_direction(iaxis) .EQ. 'TI' .OR.  
     .              line_direction(iaxis) .EQ. 'TT') THEN
               chunks(dims) = tchunk_size
	       DIMDIR = 'T'
	       dimsize = hi(idim) - lo(idim) + 1
	       IF (chunks(dims) .GT. dimsize ) GOTO 5700
	       IF (tchunk_size .GT. 0) chunk_set = 1
	    ENDIF

         ENDIF ! netcdf4

 300  CONTINUE

* If any chunk was set, then all must be set for the dimensions that are present.

      IF (chunk_set .GT. 0) THEN
         DO 350 idim = 1, nferdims
            IF (dimid(idim) .GT. 0 .AND. chunks(idim) .EQ. 0) GOTO 5600
 350     CONTINUE
      ENDIF

* put CDF into define mode so attributes can be written
      CALL CD_SET_MODE(cdfid, pcd_mode_define, status)
      IF ( status .NE. merr_ok ) RETURN

* create the new variable
      cdfstat = NF_DEF_VAR(cdfid, vname(:vlen), type, dims, dimid, varid)
      IF ( cdfstat .NE. NF_NOERR ) GOTO 5400

C     Turn on deflate_lev and chunking

      IF (netcdf4_type .EQ. 4) THEN
         IF (deflate_lev .GT. 0) CALL CD_SET_COMPRESS (cdfid, varid,
     .         deflate_lev, chunks, shuffle_flag, endian_code, status)
      ENDIF ! netcdf4

* if the "unlimited" record dimension is in use place no limits on that axis
      DO 400 idim = 1, nferdims
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
* TODO: the whole mechanism of parent grids needs developing with 6D

      IF ( itsa_slab .AND. enhead) THEN
         glen = TM_LENSTR1( gname )

         cdfstat = NF_PUT_ATT_TEXT( cdfid, varid, 'parent_grid', glen,
     .                gname(:glen) )
         cdfstat = NF_PUT_ATT_INT( cdfid, varid, 'slab_min_index', 
     .                NF_INT, 4, lo2 )
         cdfstat = NF_PUT_ATT_INT( cdfid, varid, 'slab_max_index', 
     .                NF_INT, 4, hi2 )
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

 5300  c_str = TM_FMT(FLOAT(maxstrlen), 5, 12, clen)
       CALL TM_ERRMSG
     . ( merr_linepredef, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4,
     .   'String length for variable '//vname(:vlen)//
     .   ' inconsistent w/ CDF file. On first write, use /STRLEN= at least '//
     .   c_str(:clen), no_errstring, *5900 )

 5400 CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'CD_MAKE_VAR', cdfid,
     .       unspecified_int4, 'Failed creating variable '
     .       //vname(:vlen), no_errstring, *5900 )

* error exit
 5500  vlen = TM_LENSTR1( vname )
       WRITE (buff, 1010) ww_dim_nam(idim), vname(:vlen), dlen
 1010  FORMAT ('chunk size too large for ', A,' axis of grid for ',
     .   A,' must be less than axis length written', I8)
       CALL TM_ERRMSG
     . ( merr_chunk_spec, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4, buff, no_errstring, *5900 )

 5600  CALL TM_ERRMSG
     .     ( merr_syntax, status, 'CD_MAKE_VAR', unspecified_int4, 
     .     unspecified_int4, 
     .     'If ChunkSize is set for any dimension'//
     .     ' of var, must set for all its dimensions', ' ', *5900 )

 5700  val = chunks(idim)
       c_str = TM_FMT(val, 5, 12, clen)
       val = dimsize
       d_str = TM_FMT(val, 5, 12, dlen)
       WRITE (buff, 1020) DIMDIR, c_str(1:clen), d_str(1:dlen)
 1020  FORMAT (A1,'ChunkSize ', A, 
     .         ' is larger than dimension length ', A)
       CALL TM_ERRMSG
     . ( merr_chunk_spec, status, 'CD_MAKE_VAR', unspecified_int4,
     .   unspecified_int4, buff, no_errstring, *5900 )
* error exit
 5900   RETURN
	END
