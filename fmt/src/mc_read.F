	SUBROUTINE MC_READ ( dset, ivar,
     .			a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo, use_hi, stride,
     .			var, sf_num, status )
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
* V580 9/04 *acm* - Fix bug 982 If same variable name in multiple descriptor files, 
*                   mc_init_stepfile needs to know which file its from.
* V580 10/04*acm* - Get values for scale and offset attributes in each individual 
*                   step file.  Values returned by RETURN=nc_offset or RETURN=nc_scale 
*                   will return the last value used.
* v581 *acm* 4/2005  Add sf_ndxs and sf_ndxe to use indices rather than timesteps
*                     for indexing mc datasets in mc_read.
* v581 *acm* 7/2005  Fix bug in getting start and end indices.
* V600 *acm* 8/2005  fix bug 1332, calls to  CD_GET_ATTVAL and  CD_GET_ATTVAL
*                    need cd_varid(ivar) not cd_varid as the second argument.
* V65  *acm* 1/10  Change all netcdf2-style calls and parameter names to 
*                  netcdf3-style calls, which are also consistent with netcdf4.
* V65  *acm* 1/10  Add #ifdef usingDODSf2cUnderscore, for 32-bit linux build
*                  linked with OPeNDAP library.
* V674 2/12 *acm* 6D Ferret: use nferdims rather than 4 for dimension indexing
* V71  *acm* 1/17 Ticket 2493 The E and F dimensions are "outside" of T and so 
*                 just passing a start location into CD_READ to read in each file's 
*                 of data gave incorrect results.

* include files
        include 'netcdf.inc'
        include 'tmap_errors.parm'
#include "tmap_dset.parm"
        include 'tmap_dims.parm'                ! data set dimension
#include "gt_lib.parm"
        include 'xdset_info.cmn_text'   ! data set info common
        external xdset_info_data
        include 'xstep_files.cmn_text'  ! step file common
        external xstep_files_data
        include 'xdset_info.cd_equiv'
        include 'xtm_grid.cmn_text'
        include 'xprog_state.cmn'
        include 'xmachine_int1.cmn'
        include 'xsys_names.cmn_text'


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
     .			a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .			a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .			use_lo(nferdims), use_hi(nferdims), stride(nferdims)
	REAL		var(a_lo_x:a_hi_x,
     .                      a_lo_y:a_hi_y,
     .                      a_lo_z:a_hi_z,
     .                      a_lo_t:a_hi_t,
     .                      a_lo_e:a_hi_e,
     .                      a_lo_f:a_hi_f)

! Declare netcdf function names. 
! The ones without trailing underscore are declared in netcdf.inc
#include "netcdf_declareUnderscore.inc"

* local definitions
	INTEGER*4 tmp_lo(nferdims), tmp_hi(nferdims), 
     .            start_read_lstep,end_read_lstep,
     .            idim,sfcnt,nsteps_to_read,
     .            grid, cdfstat, cdfid, l_sf1,offset,
     .            hi_offset,a_lo_t_offset,
     .            a_hi_t_offset, vartype, nvdims,vdims(8),nvatts,
     .		  varid, varpos,ndims,nvars, natts, recdim,
     .            npts, var_id,index, nsteps_in_file, vlen, end_lstep,
     .            line, m, n, k, j, i
	REAL*8	  TM_WORLD,start_tstep, end_tstep, start_read_tstep,
     .            end_read_tstep
	INTEGER   TM_SUBSC, TM_LENSTR1, prev_steps
	LOGICAL   more_to_read, TM_FPEQ, TM_DFPEQ_TOL, CD_GET_ATTVAL, 
     .            got_scale, got_off, do_warn
        PARAMETER (do_warn = .TRUE. )
        CHARACTER*2048   TM_INQ_PATH, tstep_file
	CHARACTER*128 vname
	CHARACTER*64 dim_name
	INTEGER*4 STR_CASE_BLIND_COMPARE
	REAL*8 r8buff
	CHARACTER*13    TM_STRING

	start_tstep = TM_WORLD(a_lo_t,ds_grid_number(ivar),4,box_middle)
	end_tstep   = TM_WORLD(a_hi_t,ds_grid_number(ivar),4,box_middle)
	start_read_lstep = a_lo_t
        end_lstep = a_hi_t
	start_read_tstep = start_tstep
	do 10 idim=1, nferdims
		tmp_lo(idim) = use_lo(idim)
		tmp_hi(idim) = use_hi(idim)
 10	continue

	IF (grid_line(t_dim, ds_grid_number(ivar)) .EQ. mpsnorm) THEN
* let's open the netcdf file now
*first make sure we have the full path set
          tstep_file = TM_INQ_PATH( sf_name(mc_init_stepfile(dset)), 
     .                          tmap_tsteps, ' ',
     .                          .TRUE., status )
           IF ( status .NE. merr_ok ) GOTO 5000
           sf_name(mc_init_stepfile(dset)) = tstep_file     ! save exact path
*got the full path, let's open the file now
	   cdfstat = NF_OPEN(sf_name(mc_init_stepfile(dset)), 
     .                 NF_NOWRITE, cdfid)

	   IF (cdfstat .NE. NF_NOERR) GOTO 5100
	   sf_lunit(mc_init_stepfile(dset)) = cdfid

	   cdfstat = NF_INQ_VAR(cdfid, cd_varid(ivar), vname, vartype, 
     .          nvdims, vdims, nvatts)

           vlen = TM_LENSTR1(vname)

           got_scale = CD_GET_ATTVAL(cdfid,cd_varid(ivar),'scale_factor',
     .             do_warn,vname(:vlen),cd_scale_factor(ivar),1,status )
           got_off = CD_GET_ATTVAL( cdfid, cd_varid(ivar), 'add_offset',
     .             do_warn,vname(:vlen),cd_add_offset(ivar),1,status )

*read the data
	   CALL CD_READ( dset, ivar,
     .		a_lo_x, a_lo_y, a_lo_z, a_lo_t, a_lo_e, a_lo_f,
     .		a_hi_x, a_hi_y, a_hi_z, a_hi_t, a_hi_e, a_hi_f,
     .		use_lo, use_hi, stride,
     .		var(a_lo_x, a_lo_y, a_lo_z,
     .		a_lo_t, a_hi_e, a_hi_f),
     .		mc_init_stepfile(dset), status)

* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000

*close the netcdf file
	   CALL CD_CLOSE_SET( cdfid, status)
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
*5/98 - *kob* sf_lunit needs to reflect the above cdf close
*12/00- *kob* bug - use mc_init_stepfile not sfcnt - latter not set yet
	   sf_lunit(mc_init_stepfile(dset)) = file_not_open

	ELSE
* find correct file
           prev_steps = 0
! extremely inefficient loop: should use List structure to link stepfiles
 20	   DO 30 sfcnt = 1,maxstepfiles
	      IF (sf_setnum(sfcnt) .NE. dset) GOTO 30 ! right dataset?

c	      IF ((sf_1step(sfcnt).GT. start_read_tstep) .OR. ! contains timestep
c     .		   (sf_lastep(sfcnt) .LT. start_read_tstep)) GOTO 30

! compare indices -- contains timestep
	      IF ((sf_ndxs(sfcnt).LE. start_read_lstep) .AND. ! contains timestep
     .		   (sf_ndxe(sfcnt) .GE. start_read_lstep)) GOTO 40

 30	   CONTINUE

*          NOT FOUND - fill data for this step with missing values ...
           DO n = a_lo_f, a_hi_f
             DO m = a_lo_e, a_hi_e
               DO k = a_lo_z, a_hi_z
                 DO j = a_lo_y, a_hi_y
                   DO i = a_lo_x, a_hi_x
                     var(i,j,k,start_read_lstep,m,n) = ds_bad_flag(ivar)
                   ENDDO
                 ENDDO
               ENDDO
             ENDDO
           ENDDO
*          ... and proceed to next step
           end_read_lstep = start_read_lstep
           GOTO 100

 40        CONTINUE
*          FOUND
           grid = ds_grid_number(ivar)
	   end_read_tstep = MIN(sf_lastep(sfcnt), end_tstep)
	   end_read_lstep = TM_SUBSC (end_read_tstep, grid, t_dim,
     .		a_hi_t)     !kob 11/96
	   
* correct indices, when there are multiple timesteps in sttepfiles/
           end_read_lstep = MIN(end_read_lstep, sf_ndxe(sfcnt) ) 
           end_read_lstep = MAX(start_read_lstep, end_read_lstep ) 

	   nsteps_to_read = end_read_lstep - start_read_lstep + 1
	   
* determine the index within the file to start at
c!c! uncomment this...
        grid = ds_grid_number(ivar)
	l_sf1 = TM_SUBSC (sf_1step(sfcnt), grid, t_dim,l_sf1)

* let's open the netcdf file now
*first make sure we have the full path set
* 5/99 *kob* only need to do this if the file hasn't already been opened
	   IF (.NOT. sf_valid(sfcnt)) THEN
	      tstep_file = TM_INQ_PATH( sf_name(sfcnt), tmap_tsteps, ' ',
     .                          .TRUE., status )
	      IF ( status .NE. merr_ok ) GOTO 5000
	      sf_name(sfcnt) = tstep_file ! save exact path
	   ENDIF

	   cdfstat = NF_OPEN(sf_name(sfcnt), NF_NOWRITE, cdfid) 

	   IF (cdfstat .NE. NF_NOERR) GOTO 5100
	   sf_lunit(sfcnt) = cdfid

* Verify that this is the correct variable, and if not, set it correctly
* in common..
* 2/2007 Check for success of this NF_INQ_VAR call - if the previous cd_varid
*        saved from earlier file does not exist in this stepfile, then 
*        we want to go check the other vars in the file 

	   cdfstat = NF_INQ_VAR (cdfid, cd_varid(ivar), vname, vartype, 
     .		nvdims, vdims, nvatts)
	   IF ( (cdfstat .NE. NF_NOERR) .OR. 
     .          (STR_CASE_BLIND_COMPARE ( ds_var_code(ivar), vname) 
     .		  .NE. str_match) ) THEN
	      cdfstat = NF_INQ(cdfid, ndims, nvars, natts, recdim)
	      do 50 varpos=1,nvars
* 5/99 - fix bug which was sending cd_varid(varps) to NF_INQ_VAR instead of varpos
		 cdfstat = NF_INQ_VAR (cdfid, varpos, vname, vartype, 
     .		      nvdims, vdims, nvatts)
*5/99 - bug fix: need to test to see if sting is EQUAL rather than not equal *kob*
		 IF (STR_CASE_BLIND_COMPARE ( ds_var_code(ivar), 
     .		      vname) .EQ. str_match)	THEN
		    cd_varid(ivar) = varpos
		    goto 60 
		 ENDIF
 50	      CONTINUE
	   ENDIF	
 
 60	   offset = sf_ndxs(sfcnt) - prev_steps - 1
	   nsteps_in_file = sf_ndxe(sfcnt) - sf_ndxs(sfcnt) + 1 

*calculate offset used to read data into array, if neccessary
	   a_lo_t_offset = start_read_lstep - offset
	   a_hi_t_offset = end_read_lstep - offset

* 12/00 *kob* do no sanity checking unless mode diagnostic is turned
*             on.  This necessary for dods hdf-->nc translated files
	   IF (.NOT. mode_diagnostic) THEN
	      sf_valid(sfcnt) = .TRUE.
*       now verify the timestep value in the netcdf file
*       Note this will report errors for time aggregations that have different time origins
	   ELSE
	      IF (.NOT. sf_valid(sfcnt)) THEN
* Get the T dimension and look up its attributes

		 line = grid_line(t_dim, grid)
		 dim_name = line_name_orig(line)
		 vlen = TM_LENSTR1(dim_name)

		 cdfstat = NF_INQ_VARID (cdfid, dim_name(:vlen), var_id)
* *kob* had to add an ncvinq to find out variable type of dimension because
*		 NF_FLOAT dimensions were causing core dumps. 
		 cdfstat = NF_INQ_VAR (cdfid, var_id, dim_name, vartype, nvdims, 
     .      	      vdims,nvatts)
		 CALL CD_RD_R8_1(cdfid,var_id,a_lo_t_offset,vartype,'Error',
     .      	      r8buff,status)
		 IF ( status .NE. merr_ok) GOTO 5000
		 
		 IF ( vartype .EQ. NF_DOUBLE) THEN
		    cdfstat = NF_GET_VAR1_DOUBLE(cdfid, var_id,
     .      	      a_lo_t_offset,r8buff)
		 ELSE
		 ENDIF
*       if an error is found, announce to user but continue on	
		 IF ( .NOT. TM_FPEQ(r8buff, start_read_tstep) ) THEN
		    CALL TM_ERRMSG (merr_stpmtch, status, 
     .                   'MC_READ (ignore for TSERIES aggregation)',
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

* determine local scale and offset
           vlen = TM_LENSTR1(vname)

           got_scale = CD_GET_ATTVAL( cdfid, cd_varid(ivar), 'scale_factor',
     .             do_warn,vname(:vlen),cd_scale_factor(ivar),1,status )
           got_off = CD_GET_ATTVAL( cdfid, cd_varid(ivar), 'add_offset',
     .             do_warn,vname(:vlen),cd_add_offset(ivar),1,status )

           cd_scaled(ivar) = got_scale .OR. got_scale

* let's read the data (finally). 

* (Ticket 2493) The E and F dimensions are "outside" of T and so must be read 
* sequentially if we are going to call CD_READ to do this. (Is there a more 
* efficient solution?)

	   DO n = a_lo_f, a_hi_f
	      tmp_lo(6) = n
	      tmp_hi(6) = n
	      DO m = a_lo_e, a_hi_e
	         tmp_lo(5) = m
	         tmp_hi(5) = m
	         CALL CD_READ( dset, ivar,
     .		   a_lo_x, a_lo_y, a_lo_z, a_lo_t_offset, m, n, 
     .		   a_hi_x, a_hi_y, a_hi_z, a_hi_t_offset, m, n, 
     .	           tmp_lo, tmp_hi, stride,
     .	           var(a_lo_x, a_lo_y, a_lo_z,
     .	           a_lo_t+(start_read_lstep-use_lo(4)),
     .	           m, n),
     .		   sfcnt, status)
	      ENDDO
	   ENDDO
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
	  
*close the netcdf file
	   CALL CD_CLOSE_SET( cdfid, status)
* 7/97 - *kob* error check as in tm_choose_read.F
	   if (status .NE. merr_ok) GOTO 5000
*5/98 - *kob* sf_lunit needs to reflect the above cdf close
	   sf_lunit(sfcnt) = file_not_open

* Check to see if done reading
 100       CONTINUE
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

