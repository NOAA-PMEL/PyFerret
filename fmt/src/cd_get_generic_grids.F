	SUBROUTINE CD_GET_GENERIC_GRIDS (dset,cdfid,temp_axnams,reversed,
     .					 perm, native_tax, status)

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
* Loads the common block XGRID with implicitly given grid definitions from a
* netCDF file - i.e. grid definitions for variables that lack a 
* "parent_grid = name" attribute
* also initialize grid limits (lo hi bounds on each axis) for these variables

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 2/92
* 2/95 - added support for reverse-ordered coordinate axes
* 7/96 - increase "name" from 16 to 24 characters for consistence
*	(still dimension names cannot be 24 chars --> line names in COMMON)
* 10/96 *kob* - Linux port - had to move variable assignments off of
*		declaration line.
* V500: 3/99 *sh* - if an axis doesn't exist matching the dimension name, then
*	create a 1,2,3...n axis without error message
*       4/99 *kob* - increase size of temp_axnams, vname and name to 64
* *sh*  6/99 - added "perm" (permutation) argument
* V510: *sh* 1/00 - when a "BASIC" axis is created cuz the file has no
*			coords that axis needs to bump "lastax" to get
*			checked for dup names later
*	     3/00 - convert to dynamic grids and axes
* V533: *sh* 6/01 - support for string variable input
* V552 *acm* 5/03   increase vname to 128 chars 
* V600 *ACM* attribute control. 
*             Change call to CD_GET_ATTRIB to NC_GET_ATTRIB - get attrib 
*             from linked list structure
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V683 10/12*acm* If T and F dimensions, and if both are calendar axes, insist that
*                 the calendars match.
* V685 5/13 *acm* Discrete Sampling Geometries files: if there is a variable with
*                 a cf_role attribute, assign its axis to the E direction. (If the
*                 file has coordinate variables with assigned axes, keep the 
*                 direction assigned in the file.)
* V697 11/15 *sh* return the line number of the native time axis of this dset
* V697 12/15 *am* Fix the logic around checking the calendar attributes when 
*                 grid has both T and F axes
* V710 12/16 *am* If the dimension is not a coordinate variable don't try to 
*                 check its calendar attribute
*  V75  ACM  2/2018 As in the DSG branch, pass line_direction(iaxis) to 
*                 cd_get_line_direction as an argument
*  v744 11/2011   *acm* Issue 1906: Get info for second calendar in T-F grid
*  v76   5/2020   *acm* USE/ORDER= when grids in files have different dimensions
*                       Analyze largest grid first

* argument definitions:
*       dset        - TMAP data set pointer
*	cdfid	    - netCDF id number for already opened CDF file
*       temp_axnams - nferdims axis names per grid: temp until grids are assembled
*       reversed    - logical array to record which axes have reversed coords
*       perm        - user-specified permutation (e.g. "YXZT"=2,1,3,4)
*	native_tax  - (output) line no of time axis of this dataset
*	status	    - return status

* include files

        include 'netcdf.inc'
	include 'tmap_errors.parm'
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
        include 'xbuild_grids.cmn'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xio.cmn_text'
	external xio_data
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cd_equiv'

* argument declarations
	LOGICAL  reversed(max_lines:line_ceiling)
	INTEGER	 dset, cdfid, perm(nferdims), native_tax, status
	CHARACTER*128 temp_axnams(nferdims,max_temp_grid)

* local parameter definitions:
      INTEGER     str_eq
      PARAMETER ( str_eq = 0 )

* local variable declarations:
* *kob* explicit definitions not allowed on decl line in linux...
*       moved them to parameter line 10/96
      LOGICAL NC_GET_ATTRIB, CD_GET_ATTRIB,
     .        got_it, do_warn, centers, coordvar, ok_cal, big_done,
     .        has_perm
      PARAMETER (do_warn = .TRUE. )
      PARAMETER (centers = .TRUE. )
      INTEGER TM_LENSTR1, TM_FIND_GRID_SLOT, STR_CASE_BLIND_COMPARE,
     .        STR_SAME, igrid, iaxis, grid_cnt, istat, idim,
     .        ivar, vartyp, nvdims, vdims(8), nvatts, vlen, idim2,
     .        tmvar, tmpgrid, npts, dim, tmplines(nferdims), ndim,
     .	      grid_dims(nferdims), trans(nferdims), indices(nferdims),
     .        i, maxlen, attlen,attoutflag, all_outflag, axid, iaxis6,
     .        attid, eax, cfrole_dim, blen, max_dims, big_var
	
      REAL    tmp
      CHARACTER name*128, vname*128, linunits*24, 
     .        cal_str*24, cal_t*24, cal_f*24, dsave*2

      INTEGER tt  ! nice short name
      EQUIVALENCE (tt,lunit_errors)

* initialize
	native_tax = mpsnorm

* initialize temporary grid name pointer
	num_tmp_grids = 0
	DO 5 i = max_grids, grid_ceiling
 5	   tmp_nam_ptr(i) = 0

* * * * loop through each variable in this TMAP data set * * *
* Check for cf_role attribute. If present, its grid is in the E direction.

	cfrole_dim = 0 

* How many dimensions do the varibles have? Make a grid for the first one
* encountered that has the highest number of dimensions first.

	max_dims = 0
	big_var = 0
	
	DO tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) CYCLE
           ivar = cd_varid(tmvar)
           CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)
	   IF (nvdims .GT. max_dims) big_var = ivar
	   max_dims = MAX(max_dims, nvdims)
	ENDDO 
	
	big_done = .FALSE.

        DO 25 tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 25
	   eax = 0
           ivar = cd_varid(tmvar)

	   IF (.NOT.big_done .AND. ivar.NE.big_var) CYCLE
	   IF (ivar .EQ. big_var) big_done = .TRUE.

* get the vital statistics for this variable
           CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)

           IF (istat.NE.merr_ok) CALL TM_ERRMSG
     .     ( istat+pcdferr,status,'CD_GET_GENERIC_GRIDS',cdfid,ivar,
     .       no_errstring, no_errstring, *5900 )
           vlen = TM_LENSTR1(vname)

* if a string variable, the first dimension is the string axis (ignore it)
	   IF (vartyp .EQ. NF_CHAR) THEN
	      nvdims = nvdims - 1
	      DO idim = 1, nvdims
	         vdims(idim) = vdims(idim+1)
	      ENDDO
	   ENDIF

* if it has a cf_role attribute, we'll put its axis in the E direction.
*  trajectory:cf_role = "trajectory_id" (or var w/ "profile_id", or "timeseries_id")
* (IF there is a instance_dimension attribute, it's value is the the instance axis,
* to be put in the E direction. If there is a sample_dimension attribute on some variable, 
* its value is the obs  dimension, to be put in the X direction.) 
* All that is true for Point, Timeseries, Trajectory or Profile data.  If we have a 
* TrajectoryProfile or TimeseriesProfile dataset, the instance_dimension attribute points 
* to the trajectory-index or timeseries-index dimension. For this reason think of the 
* instance dimension for point, trajectory, timeseries, or profilefeature types as the 
* ifeature-dim.  For the compound feature types, ifeature-dim is the dimension counting 
* number of profiles.

	   has_perm = .FALSE.
	   DO idim=1, nferdims
	      IF (perm(idim) .NE. unspecified_int4) has_perm = .TRUE.
	   ENDDO

	   CALL  CD_GET_VAR_ATT_ID (dset, ivar, "cf_role", attid, status)
	   IF (status .EQ. merr_ok .AND. .NOT.has_perm) THEN
	      eax = vdims(1) 
	      dsg_feature_var(dset) = ivar

* find the instance dimension

              DO 15 idim = 1, nvdims
	         IF (vdims(idim) .EQ. eax) cfrole_dim = idim
 15          CONTINUE
             GOTO 35  ! only one cf_role attribute recognized

	   ENDIF

 25       CONTINUE

 35       CONTINUE

* * * * loop through each variable in this TMAP data set * * *
* if it doesn't have a grid pointer create a suitable one and point to it
        DO 500 tmvar = 1,maxvars
           IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 500
           ivar = cd_varid(tmvar)

* does it point explicitly to a parent grid ?
          maxlen = 24   !linenamlen
          got_it = NC_GET_ATTRIB( dset, ivar, 'parent_grid',
     .                       .NOT.do_warn, ' ', maxlen, attlen, 
     .                       attoutflag, name, tmp )
          IF ( got_it ) GOTO 500  ! yes - already has a pointer

* get the vital statistics for this variable
           CALL CD_GET_VAR_INFO(dset, ivar, vname, vartyp, nvdims,
     .                 vdims, nvatts, coordvar, all_outflag, istat)

           IF (istat.NE.merr_ok) CALL TM_ERRMSG
     .     ( istat+pcdferr,status,'CD_GET_GENERIC_GRIDS',cdfid,ivar,
     .       no_errstring, no_errstring, *5900 )
           vlen = TM_LENSTR1(vname)

* if a string variable, the first dimension is the string axis (ignore it)
	   IF (vartyp .EQ. NF_CHAR) THEN
	      nvdims = nvdims - 1
	      DO idim = 1, nvdims
	         vdims(idim) = vdims(idim+1)
	      ENDDO
	   ENDIF

* use the next grid building slot to build a grid for this variable
           tmpgrid = num_tmp_grids + 1

* initialize the axis names in case there are fewer than nferdims from CDF variable
           DO 10 idim = 1, nferdims
              tmplines(idim) = unspecified_int4
              temp_axnams(idim,tmpgrid)='NORMAL'
 10        CONTINUE

* find the axis names from the CDF file
	   cal_t = ' '
	   cal_f = ' '
           DO 200 idim = 1, nvdims

* ... get name of dimension
              CALL CD_GET_DS_DIMS( dset, vdims(idim), name, npts, istat )
              IF (istat.NE.merr_ok) CALL TM_ERRMSG
     .     ( istat+pcdferr,status,'CD_GET_GENERIC_GRIDS',cdfid,ivar,
     .       vname(:vlen), 'cant get dimensions', *5900 )
 
* ... locate the axis by that name (processed in CD_GET_AXES)
	      iaxis = 0
 100	      CALL TM_NEXT_TMP_LINE(iaxis, *120)  ! 120 if no more
	         istat = STR_CASE_BLIND_COMPARE(name, line_name(iaxis))
	         IF ( istat .EQ. str_eq ) GOTO 150  ! got a match
	      GOTO 100

* ... axis doesn't exist - perhaps it is a dimension with no variable defn
* *sh* 3/99: or perhaps a 2D variable with name matching the dimension name
*  either way -- not an error condition if we need to synthesize an axis ...
 120	      CALL TM_MAKE_BASIC_AXIS( name, 1.0D0, 1.0D0, npts,
     .                                 iaxis, status )
              IF ( status .NE. merr_ok ) GOTO 5900

* Are there attributes that tell us the direction?  If so apply them.
* for the user's convenience, an invalid T axis is still in the T direction.
* The routine will get and use basic info about the units.

* axis ID
	      CALL CD_GET_VAR_ID (dset, name, axid, status)
	      linunits = char_init24
	      IF (idim .EQ. cfrole_dim) THEN
	         line_direction(iaxis) = 'EE'
		 cfrole_dim = 0  ! reset. Just one axis will have this role.
		 eax = 0
	      ELSE
	         dsave = line_direction(iaxis)
		 CALL CD_GET_LINE_DIRECTION (dset, axid, name, linunits, 
     .                do_warn, .TRUE., line_direction(iaxis))
		 IF (has_perm .AND. line_direction(iaxis) .EQ. 'EE') 
     .                line_direction(iaxis) = dsave
	      ENDIF

* save a pointer to the axis just found
 150          IF (idim.LE.nferdims) tmplines(idim) = iaxis

              CALL CD_GET_VAR_ID (dset, name, axid, status)

* Check the calendar attribute 
	      IF (status .EQ. merr_ok) THEN 
	         blen = 24
                 got_it = CD_GET_ATTRIB( cdfid, axid, 'calendar',
     .                  do_warn, name, cal_str, attlen, blen )
	         IF (got_it .AND. line_direction(iaxis)(1:1).EQ.'T') THEN
		    cal_t = cal_str
		    CALL CD_GET_CALENDAR_NAME(cal_str, ok_cal)
		    IF (ok_cal) line_cal_name(iaxis) = cal_str
		 ENDIF
	         IF (got_it .AND. line_direction(iaxis)(1:1).EQ.'F') THEN
		    cal_f = cal_str
		    CALL CD_GET_CALENDAR_NAME(cal_str, ok_cal)
		    IF (ok_cal) line_cal_name(iaxis) = cal_str
		 ENDIF
	      ENDIF

 200       CONTINUE

* reorder the axes into the most likely order if not explicitly given
           IF ( ds_ordering(1,tmvar) .EQ. unspecified_int4 ) THEN
              CALL TM_AXIS_ORDER(tmplines, perm, ds_ordering(1,tmvar),
     .                           nvdims, vname(:vlen))
           ENDIF

* ... make a sorted list of the relevant axes in grid_dims
*     and a list of the index ordering of the axes in indices
	ndim = 0
	DO 210 idim = 1, nferdims
	   trans(idim) =  ABS(ds_ordering(idim,tmvar))
	   grid_dims(idim) = trans(idim)
	   indices(idim) = idim
	   IF (trans(idim) .NE. 0 ) THEN
	      ndim = ndim + 1
	   ENDIF
 210	CONTINUE
	DO 220 idim = 1, ndim
	   DO 215 idim2 = idim+1, ndim
	      IF (grid_dims(idim) .GT. grid_dims(idim2)) THEN
	         dim = grid_dims(idim)
	         grid_dims(idim ) = grid_dims(idim2)
	         grid_dims(idim2) = dim
	      ENDIF
 215	   CONTINUE
 220	CONTINUE
	DO 230 idim = 1, ndim
	   i = 1
	   DO 225 idim2 = 1, ndim
	      IF (trans(idim2) .LT. trans(idim)) i = i + 1
 225	   CONTINUE
	   indices(idim) = i
 230	CONTINUE

* save the axis names in the proper order
           DO 240 idim = 1, nferdims
              ds_grid_start(idim,tmvar) = 1
              ds_grid_end  (idim,tmvar) = 1
 240       CONTINUE
           DO 250 idim = 1, ndim
              dim = trans(idim)
              iaxis = tmplines(indices(idim))
	      idim2 = grid_dims(idim)
              temp_axnams(idim2,tmpgrid) = line_name(iaxis)
              ds_grid_end(idim2,tmvar)   = line_dim(iaxis)
* ... arbitrarily, the native taxis taken from the first grid with a time axis
	      IF (idim2.EQ.t_dim .AND. native_tax.EQ.mpsnorm)
     .						native_tax = iaxis
* ... flag reverse-ordered coordinate axis of variable (2/95)
	      IF ( reversed(iaxis) ) ds_ordering(idim,tmvar)
     .			      = -1 * ds_ordering(idim,tmvar)
 250       CONTINUE

* ... If there is a T and an F axis, check for matching calendars

	   iaxis = 0
	   iaxis6 = 0
           DO 260 idim = 1, ndim
	      IF (trans(idim) .EQ. T_DIM) iaxis  = tmplines(idim)
	      IF (trans(idim) .EQ. F_DIM) iaxis6 = tmplines(idim)
 260       CONTINUE

*     When there is a calendar attrib on the the F or T axis but not both, 
*     use the specified calendar for the both time dimensions

	   IF (iaxis.GT.0 .AND. iaxis6.GT.0) THEN
	      IF ( (STR_SAME(cal_f, ' ') .EQ. 0) )
     .           line_cal_name(iaxis6) = line_cal_name(iaxis)
	      IF ( (STR_SAME(cal_t, ' ') .EQ. 0) )
     .           line_cal_name(iaxis) = line_cal_name(iaxis6)

              IF (STR_SAME(line_cal_name(iaxis), line_cal_name(iaxis6)) .NE. 0) 		
     .		CALL TM_ERRMSG (merr_axis_att, status, 'CD_GET_GENERIC_GRIDS',
     .            no_descfile, 1,
     .		  'Calendar definitions on T and F axes must match',
     .            no_errstring, *5900 )
	   ENDIF

* is the temporary grid we just created unique ?
	   igrid = 0
 300	   CALL TM_NEXT_TMP_GRID( igrid, *320)
	      grid_cnt = tmp_nam_ptr(igrid)
              DO 310 idim = 1, nferdims
                 IF ( temp_axnams(idim,tmpgrid )
     .          .NE.  temp_axnams(idim,grid_cnt) ) GOTO 300
 310          CONTINUE
* ... not unique - exact match with one that already exists
              GOTO 350
 320	   CONTINUE
 

* allocate a new grid
* find next location to store info; if too many grids - give error
	  CALL TM_ALLO_TMP_GRID(igrid, status)
	  IF (status .NE. merr_ok) GOTO 5900

* make up a unique grid name "GXXnn" where XX is hashed from data set name
          grid_name(igrid) = 'G'//cd_dset_code(dset)(:2)
          CALL TM_NEW_GRID_NAME( grid_name(igrid), name )
          grid_name(igrid) = name

* basic defaults
          grid_rotation(igrid) = 0.0
          DO 340 idim = 1, nferdims
 340      grid_out_prod(idim,igrid) = .TRUE.

* save a pointer to this grid for the TMAP variable
 350      ds_grid_number(tmvar) = igrid

 500   CONTINUE

* successful completion
 	status = merr_ok
        RETURN

* error exit
 5900	RETURN

	END
