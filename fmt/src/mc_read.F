	SUBROUTINE MC_READ ( dset, ivar,
     .                       a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .                       a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			     use_lo, use_hi, stride,
     .			     var, sf_num, status )
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
*
* Reads a 4D block of data (lower D being a degenerate case with dimensions
* being 1 point thick) from a multi-file netCDF dataset
*
* written 8/95 by kob for pmel/tmap
* 11/96 *kob* - Linux Port
* 	      - replace end_read_tstep with a_hi_t as a clue for tm_subsc
* 7/97 *kob* - fix bug in which status is not being checked
* 5/98 *kob* - fix bug in which step files were not being properly reset after 
*              they were closed
* V500 5/99 *kob* - 1) fixed bug occurring when correctly setting up variable
*	               in common
*                   2) minor mod to only open set entire path of cdf file if it
*                      hasn't already been opened
*           *sh* - added strides to CD_READ call
* V510 *sh* 3/00 - increase allowable path name length
* v522 *kob* 12/00 - added xprog_state common file in order to check for
*                    mode diagnostic.  only do sanity checking on step
*                    files if mode diag is turned on.
* v531: *kob* 5/01 - fix erroneous error reporting at label 5100
* V552 5/03 *acm*: increase size of vname to 128 chars
* V570 4/04 *acm* when checking which stepfile to read from, use
*                 TM_FPEQ_TOL to see if the timestep is equal (within tolerance)
*                 to the start or end point of any of the files.

* arguments:
*	dset        - internal data set # (from GT_INIT_DSET)
*	ivar        - position in common of variable information
*       actual_dims - "a_" full dimensions of data_grid array passed
*	use_lo      - low indices of model data block desired
*	use_hi      - high indices of model data block desired
*	var         - array containing model data (output)
*	sf_num      - stepfile number (returned from GT_FIND_TSTEP (output)
*	status      - error status upon completion (output)
*
	INTEGER		dset, ivar, sf_num, status, ndim,
     .                  a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .                  a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .			use_lo(4), use_hi(4), stride(4)
	REAL		var(a_lo_x:a_hi_x,
     .                      a_lo_y:a_hi_y,
     .                      a_lo_z:a_hi_z,
     .                      a_lo_t:a_hi_t)

* include files
        include 'netcdf.decl'
#include "netcdf.inc"
	include 'tmap_errors.parm'
#include "tmap_dset.parm"
	include 'tmap_dims.parm'		! data set dimension
#include "gt_lib.parm"
	include 'xdset_info.cmn_text'	! data set info common
	external xdset_info_data
	include 'xstep_files.cmn_text'	! step file common
	external xstep_files_data
        include 'xdset_info.cd_equiv'
	include 'xtm_grid.cmn_text'
	include 'xprog_state.cmn'
#ifdef sun
        include 'xmachine_byte.cmn'
#else
        include 'xmachine_int1.cmn'
#endif
	include 'xsys_names.cmn_text'


* local definitions
	INTEGER*4 tmp_lo(4), tmp_hi(4), start_read_lstep,end_read_lstep,
     .            idim,sfcnt,nsteps_to_read,
     .            grid, cdfstat, cdfid, l_sf1,offset,
     .            hi_offset,a_lo_t_offset,
     .            a_hi_t_offset, vartype, nvdims,vdims(8),nvatts,
     .		  varid, varpos,ndims,nvars, natts, recdim,
     .            npts, var_id,index,nsteps_in_file
	REAL*8	  TM_WORLD,start_tstep, end_tstep, start_read_tstep,
     .            end_read_tstep, epsilon
	INTEGER   TM_SUBSC, NCOPN, NCVID
	LOGICAL   more_to_read, TM_FPEQ, TM_DFPEQ_TOL
        CHARACTER*256   TM_INQ_PATH, tstep_file
	CHARACTER*128 vname
	CHARACTER*64 dim_name
	INTEGER*4 STR_CASE_BLIND_COMPARE
	REAL*8 r8buff
	CHARACTER*13    TM_STRING

        epsilon = 4.E-15   ! for TM_DFPEQ_TOL

	start_tstep = TM_WORLD(a_lo_t,ds_grid_number(ivar),4,box_middle)
	end_tstep   = TM_WORLD(a_hi_t,ds_grid_number(ivar),4,box_middle)
	start_read_lstep = a_lo_t
	start_read_tstep = start_tstep
	do 10 idim=1,3
		tmp_lo(idim) = use_lo(idim)
		tmp_hi(idim) = use_hi(idim)
 10	continue

	IF (grid_line(t_dim, ds_grid_number(ivar)) .EQ. mpsnorm) THEN
* let's open the netcdf file now
*first make sure we have the full path set
          tstep_file = TM_INQ_PATH( sf_name(mc_init_stepfile), 
     .                          tmap_tsteps, ' ',
     .                          .TRUE., status )
           IF ( status .NE. merr_ok ) GOTO 5000
           sf_name(mc_init_stepfile) = tstep_file     ! save exact path
*got the full path, let's open the file now
	   cdfid = NCOPN(sf_name(mc_init_stepfile), NCNOWRIT, cdfstat)	
	   IF (cdfstat .NE. NCNOERR) GOTO 5100
	   sf_lunit(mc_init_stepfile) = cdfid
*read the data
	   CALL CD_READ( dset, ivar,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t,
     .		use_lo, use_hi, stride,
     .		var(a_lo_x, a_lo_y, a_lo_z,
     .		a_lo_t),
     .		mc_init_stepfile, status)

* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000

*close the netcdf file
	   CALL CD_CLOSE_SET( cdfid, status)
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
*5/98 - *kob* sf_lunit needs to reflect the above cdf close
*12/00- *kob* bug - use mc_init_stepfile not sfcnt - latter not set yet
	   sf_lunit(mc_init_stepfile) = file_not_open

	


	ELSE
* find correct file
 20	   DO 30 sfcnt = 1,maxstepfiles
	      IF (sf_setnum(sfcnt) .NE. dset) GOTO 30 ! right dataset?

c	      IF ((sf_1step(sfcnt).GT. start_read_tstep) .OR. ! contains timestep
c     .		   (sf_lastep(sfcnt) .LT. start_read_tstep)) GOTO 30

          ! contains timestep, to within epsilon
	      IF (TM_DFPEQ_TOL(sf_1step(sfcnt), 
     .                         start_read_tstep, epsilon) .OR.
     .		  TM_DFPEQ_TOL(sf_lastep(sfcnt), 
     .                         start_read_tstep, epsilon)) GOTO 40

          ! stepfile does not contain timestep
	      IF ((sf_1step(sfcnt).GT. start_read_tstep) .OR. 
     .		   (sf_lastep(sfcnt) .LT. start_read_tstep)) GOTO 30
	      GOTO 40
 30	   CONTINUE
*       found it
	   
 40	   grid = ds_grid_number(ivar)
	   end_read_tstep = MIN(sf_lastep(sfcnt), end_tstep)
	   end_read_lstep = TM_SUBSC (end_read_tstep, grid, t_dim,
     .		a_hi_t)     !kob 11/96
	   
	   nsteps_to_read = end_read_lstep - start_read_lstep + 1
	   
	   
* determine the index within the file to start at
*	grid = ds_grid_number(ivar)
*	l_sf1 = TM_SUBSC (sf_1step(sfcnt), grid, t_dim,l_sf1)
	

* let's open the netcdf file now
*first make sure we have the full path set
* 5/99 *kob* only need to do this if the file hasn't already been opened
	   IF (.NOT. sf_valid(sfcnt)) THEN
	      tstep_file = TM_INQ_PATH( sf_name(sfcnt), tmap_tsteps, ' ',
     .                          .TRUE., status )
	      IF ( status .NE. merr_ok ) GOTO 5000
	      sf_name(sfcnt) = tstep_file ! save exact path
	   ENDIF

	   cdfid = NCOPN(sf_name(sfcnt), NCNOWRIT, cdfstat)	
	   IF (cdfstat .NE. NCNOERR) GOTO 5100
	   sf_lunit(sfcnt) = cdfid



* Verify that this is the correct variable, and if not, set it correctly
* in common..
	   CALL NCVINQ(cdfid, cd_varid(ivar), vname, vartype, nvdims, vdims,
     .		nvatts,cdfstat)
	   IF (STR_CASE_BLIND_COMPARE ( ds_var_code(ivar), 
     .		vname) .NE. str_match)	THEN
	      CALL NCINQ(cdfid, ndims, nvars, natts, recdim, cdfstat)
	      do 50 varpos=1,nvars
* 5/99 - fix bug which was sending cd_varid(varps) to NCVINQ instead of varpos
		 CALL NCVINQ(cdfid, varpos, vname, vartype, 
     .		      nvdims, vdims, nvatts,cdfstat)
*5/99 - bug fix: need to test to see if sting is EQUAL rather than not equal *kob*
		 IF (STR_CASE_BLIND_COMPARE ( ds_var_code(ivar), 
     .		      vname) .EQ. str_match)	THEN
		    cd_varid(ivar) = varpos
		    goto 60 
		 ENDIF
 50	      CONTINUE
	   ENDIF	
 

 60	   offset = TM_SUBSC (sf_1step(sfcnt), grid, t_dim,l_sf1) - 1
	   nsteps_in_file = TM_SUBSC (sf_lastep(sfcnt), grid, t_dim,l_sf1) -
     .		TM_SUBSC (sf_1step(sfcnt), grid, t_dim,l_sf1) + 1 
	   
*calculate offset used to read data into array, if neccessary
	   a_lo_t_offset = start_read_lstep - offset
	   a_hi_t_offset = end_read_lstep - offset


* 12/00 *kob* do no sanity checking unless mode diagnostic is turned
*             on.  This necessary for dods hdf-->nc translated files
	   IF (.NOT. mode_diagnostic) THEN
	      sf_valid(sfcnt) = .TRUE.
*       now verify the timestep value in the netcdf file
	   ELSE
	      IF (.NOT. sf_valid(sfcnt)) THEN
		 CALL NCDINQ( cdfid, vdims(nvdims), dim_name, 
     .                           npts, cdfstat )
		 var_id = NCVID(cdfid, dim_name, cdfstat)
* *kob* had to add an ncvinq to find out variable type of dimension because
*		 NCFLOAT dimensions were causing core dumps. 
		 CALL NCVINQ(cdfid, var_id, dim_name, vartype, nvdims, 
     .      	      vdims,nvatts,cdfstat)
		 CALL CD_RD_R8_1(cdfid,var_id,a_lo_t_offset,vartype,'Error',
     .      	      r8buff,status)
		 IF ( status .NE. merr_ok) GOTO 5000
		 
		 IF ( vartype .EQ. NCDOUBLE) THEN
		    CALL NCVGT1(cdfid, var_id,a_lo_t_offset,r8buff,cdfstat)
		 ELSE
		 ENDIF
*       if an error is found, announce to user but continue on	
		 IF (.NOT. TM_FPEQ(SNGL(r8buff), 
     .                        SNGL(start_read_tstep))) THEN
		    CALL TM_ERRMSG (merr_stpmtch, status, 'MC_READ',
     .           	 sf_setnum(sfcnt), sfcnt,
     .  		 TM_STRING(start_read_tstep), TM_STRING(r8buff),*70)
*Now check to make sure that the number of files in this stepfile pulled 
*from the descriptor agree with the number of points on the time axis
		 ELSE IF ( npts .NE. nsteps_in_file) THEN
		    sf_valid(sfcnt) = .FALSE.
		 ELSE
		    sf_valid(sfcnt) = .TRUE.
		 ENDIF
	      ENDIF
	   ENDIF
* determine local start and end indices
 70	   tmp_lo(4) = start_read_lstep -offset
	   tmp_hi(4) = tmp_lo(4) + (end_read_lstep - start_read_lstep)
		

* let's read the data (finally)
	   CALL CD_READ( dset, ivar,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t_offset,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t_offset,
     .	        tmp_lo, tmp_hi, stride,
     .	        var(a_lo_x, a_lo_y, a_lo_z,
     .	        a_lo_t+(start_read_lstep-use_lo(4))),
     .		sfcnt, status)
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
	  
*close the netcdf file
	   CALL CD_CLOSE_SET( cdfid, status)
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
*5/98 - *kob* sf_lunit needs to reflect the above cdf close
	   sf_lunit(sfcnt) = file_not_open

* Check to see if done reading
	   IF (end_read_lstep .LT. a_hi_t) THEN
	      start_read_lstep = end_read_lstep + 1
	      start_read_tstep = TM_WORLD(start_read_lstep,
     .        ds_grid_number(ivar),4,box_middle)
	      GOTO 20 
	   ENDIF
	   
	ENDIF
 200	RETURN

 5100  CALL TM_ERRMSG
     .     ( cdfstat+pcdferr, status, 'MC_READ', unspecified_int4,
     .     no_varid, no_errstring,
     .     no_errstring, *5000 )


 5000	RETURN


	END

