	SUBROUTINE TM_MAKE_4D_GRIDS ( dnum, status )

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
* Will check a dataset to make sure it has no mpsunkn axis. If it has an
* unknown time axis, it will create an axis using %%_GET_TIME_AXIS, then
* construct a 4D grid from that. If it has other axis with mpsunkn, it will
* issue an error.
*
* written 07/88 by M. Verschell for PMEL/TMAP
*
* revision 0.00 - 07/06/88
* revision 0.10 - 07/28/88 - If GT data set calls GT_GET_TIME_AXIS, if TS data
*			     set it calls TS_GET_TIME_AXIS.
* revision 0.11 - 11/21/88 - is var used check changed to specific char_initXX
* revision 0.12 - 11/29/88 - munknown changed to mpsunkn in grid_line check
* revision 0.13 - 02/08/89 - added ds_time_modulo sets line_modulo on time axis
* revision 0.20 -  9/12/90 - *sh* make auto time axis only if needed
* Unix/RISC port - 3/12/91 *sh* - fixed bug: variable "new" not initialized
* 6/91 - kob - Time independent data handling capability added.
*            -  proc's STR_UPCASE and INDEX used to see if the
*               word INDEPENDENT exists in t0_time, if so the
*               the data is considered time independent
* 12/92 - *sh* - set grid_out_prod when inserting time axis
* 1/93  - *sh* - check for "-1:-1" limits given for an explicit T axis
*
* 7/95  - *kob*- added logic to add a time axis to a multi-file netcdf data
*		 set based on the information in a descriptor

* V510 5/00 *sh* - adapt to dynamic grids and axes for netCDF files
*      6/00 *sh* bug fix: T axis ds_grid_start/end set only on first var using
*		each grid
* V530 1/00 *kob* - mods for aggregating time_independent files in an MC
*                   context
* V541 2/02 *sh* - minor change to support line_modulo_len
* 4/03 *acm* - Use STR_SAME to compare strings
*       *acm* 3/12 Add E and F dimensions (use nferdims in tmap_dims.parm)
*       *acm* 3/12 removing use of old GT and TS data formats (routine 
*	           GT_GET_TIME_AXIS is still used for MC datasets).
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 

* CALLING ARGUMENTS
*	dnum - the data set fow which a time axis is desired
*	status  - the returning status
	INTEGER*4	dnum, status
*
* INCLUDE FILES
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	include 'xstep_files.cmn_text'
	external xstep_files_data

*
* LOCAL DEFINITIONS
* upstat, temp_t0time,STR_UPCASE added for implementation of time-independent
* data ------------------> kob 6/91
        CHARACTER*11    temp_t0time               ! kob-
        INTEGER*4       upstat                    ! 6/91
	LOGICAL		use_ext_tax
	CHARACTER*13    TM_STRING
	INTEGER*4	ivar, iv, ngrd, ilne, ngrid, ext_tax, sf
	INTEGER*4	GT_GET_TIME_AXIS, 
     .			TM_FIND_LIKE_GRID, STR_UPCASE
* local parameter declarations:
        INTEGER*4 STR_SAME, new
        PARAMETER ( new = 0 )
	LOGICAL all_normal

* initialize
	use_ext_tax = .TRUE.	! assume auto axis isn't needed

* *kob* 12/00 - loop through all variables and check them all 
*               for time indepedence.   
	all_normal = .TRUE.
	DO 50 ivar = 1,maxvars
	   ngrd = ds_grid_number(ivar)
	   IF (ds_var_setnum(ivar) .EQ. dnum) THEN
	      IF ( grid_line(4,ngrd) .NE. mpsnorm) THEN
		 all_normal = .FALSE.
		 GOTO 70
	      ENDIF
	   ENDIF
 50	CONTINUE

*       Loop through all variables
 70	DO 150 ivar = 1,maxvars
	  IF (ds_var_setnum(ivar) .EQ. dnum) THEN

* . Are any non-time axes defined as mpsunkn ?
	    ngrd = ds_grid_number(ivar)
	    DO 100 ilne = 1,3
	      IF ( grid_line(ilne,ngrd) .EQ. mpsunkn ) GOTO 9100
  100	    CONTINUE

* .. If time axis unknown create an auto time axis (once, only)
*	and build a new nferdims-D grid in position 0 with time info
* *kob* 12/00 also do this if all the variables are time independent
	    IF ( grid_line(4,ngrd) .EQ. mpsunkn .OR. 
     .             ( (STR_SAME(ds_type(dnum), '  MC') .EQ. 0) .AND. 
     .              grid_line(4,ngrd) .NE. mpsnorm) .OR.
     .	            (STR_SAME(ds_type(dnum), '  MC') .eq. 0) .AND.
     .		    all_normal ) THEN
* ... create a time axis from the descriptor information (*sh* 9/90)
* ... IF statement added to check for time-independent data ----> kob 6/91
* ...   and to make sure time_axis is set to normal for time independent data
	      IF ( use_ext_tax ) THEN
	         use_ext_tax = .FALSE.
                 upstat = STR_UPCASE(temp_t0time,ds_t0time(dnum))
                 IF (INDEX(temp_t0time,'INDEPENDENT') .GT. 0) THEN   ! kob -
                    ds_time_axis(dnum) = mpsnorm                     ! 6/91
	         ELSEIF ((STR_SAME(ds_type(dnum), '  GT') .EQ. 0) .OR. 
     .                   (STR_SAME(ds_type(dnum), '  MC') .EQ. 0) ) THEN
	           ds_time_axis(dnum) = GT_GET_TIME_AXIS(dnum,status)
	         ELSE
	           GOTO 9200
	         ENDIF
	         IF (status .NE. merr_ok) GOTO 9999
* Do not set line modulo if data is time-independent ----------> kob 6/91
	         IF (ds_time_modulo(dnum) .AND. (INDEX(temp_t0time,
     .                'INDEPENDENT') .EQ. 0))
     .			     line_modulo(ds_time_axis(dnum)) = .TRUE.
	         line_modulo_len(ds_time_axis(dnum)) = 0.0D0
	      ENDIF

* ... Transfer things to position 0 ("new") from old 3D grid
	      CALL TM_COPY_GRID(ngrd,new)
	      grid_line(4,new) = ds_time_axis(dnum)
              grid_out_prod(4,new) = .TRUE.
* ... Does this match an already existing grid?
	      ngrid = TM_FIND_LIKE_GRID ( new )
	      IF (ngrid .EQ. unspecified_int4) THEN
*	... This is a new grid definition -- make sure name is unique
	         grid_name(new) = grid_name(ngrd)
	         grid_name(ngrd) = '%noexist%'
	         CALL TM_NEW_GRID_NAME(grid_name(new), grid_name(new))
	         grid_name(ngrd) = grid_name(new)
*	... Keep the new grid with the time axis and "use" the time axis
*	    Other vars defined on this grid will then bypass this code
	         CALL TM_COPY_GRID( new, ngrd )
	         CALL TM_USE_LINE( ds_time_axis(dnum) )
	      ELSE
*	... This duplicates a pre-existing grid definition -- reuse the older
	         DO 110 iv =  1,maxvars
	           IF (ds_var_setnum(ivar) .EQ. dnum) THEN
	             IF (ds_grid_number(iv).EQ.ngrd)
     .			 ds_grid_number(iv) = ngrid
	           ENDIF
 110		 CONTINUE
	         ngrd = ngrid
	      ENDIF
* ... Set the (previously unspecified) subscript limits for the variable
* ... Check to see if data is time-independent, if it is, both
* ...... grid start and end are set to 1 -----------> kob 6/91
	      DO 120 iv =  1,maxvars
	        IF (ds_var_setnum(ivar).EQ.dnum
     .	     .AND.  ds_grid_number(iv) .EQ.ngrd  ) THEN
	          ds_grid_start(4,iv) = 1
                  IF (INDEX(temp_t0time,'INDEPENDENT') .GT. 0) THEN
                    ds_grid_end(4,iv) = 1
                  ELSE
                    ds_grid_end  (4,iv) = line_dim( ds_time_axis(dnum) )
                  ENDIF
	        ENDIF
 120	      CONTINUE
	    ELSE
	      ext_tax = grid_line(4,ngrd)	! save external time ax #
	    ENDIF

	  ENDIF

  150	CONTINUE

* a few clean-up details for externally defined time axis (*sh* 9/90)
* (uses time axis from last data set variable as data set default)
	IF ( use_ext_tax ) THEN
	   ds_time_axis  ( dnum ) = ext_tax
	   ds_time_modulo( dnum ) = line_modulo( ext_tax )
	   DO 160 sf = 1, maxstepfiles
 160	   IF ( sf_setnum(sf) .EQ. dnum ) sf_regtimes(sf) = .FALSE.
* *sh* 1/93: check for "-1:-1" limits with an explicit time axis
	   DO 170 ivar = 1,maxvars
	      IF (ds_var_setnum(ivar) .EQ. dnum) THEN
                 IF ( ds_grid_start(4,ivar) .LE. 0 )
     .                 ds_grid_start(4,ivar) = 1
                 IF ( ds_grid_end(4,ivar) .LE. 0 )
     .                 ds_grid_end(4,ivar) =  line_dim( ext_tax )
              ENDIF
 170       CONTINUE
	ENDIF

	GOTO 9990

* error messages
 9000	CALL TM_ERRMSG (merr_gridlim, status, 'TM_MAKE_4D_GRIDS',
     .			no_descfile, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(max_grids)), no_errstring,
     .			*9999)
 9100	CALL TM_ERRMSG (merr_undefline, status, 'TM_MAKE_4D_GRIDS',
     .			dnum, no_stepfile,
     .			'VARIABLE = '//ds_var_code(ivar), no_errstring,
     .			*9999)
 9200	CALL TM_ERRMSG (merr_unktype, status, 'TM_MAKE_4D_GRIDS',
     .			dnum, no_stepfile,
     .			'Type declared '//ds_type(dnum),
     .			no_errstring, *9999)
 9990	status = merr_ok

 9999	RETURN
	END
