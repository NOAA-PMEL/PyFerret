	SUBROUTINE CD_GET_TIME_AXIS(dset, recdim, status)

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
* and save its lo/hi limits in stepfile common
* This is needed for compatibility with GT and TS formats

* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* revision 0.0 - 3/92
*
* 7/95 *kob* - added check for MC dataset to prevent clobbering of
*	       stepfile time data already stored


* argument definitions:
*       dset        - TMAP data set pointer
*       recdim      - orientation of "time axis" (usually t_dim)
*	status	    - return status

* include files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'tmap_format/xtm_grid.cmn_text'
	external xgt_grid_data
        include 'tmap_format/xdset_info.cd_equiv'
	include 'tmap_format/xstep_files.cmn_text'
	external xstep_files_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
        INCLUDE 'TMAP_FORMAT:XDSET_INFO.CD_EQUIV'
        INCLUDE 'TMAP_FORMAT:XSTEP_FILES.CMN'
#endif

* argument declarations
	INTEGER	 dset, recdim, status

* local variable declarations:
      REAL*8  TM_WORLD
      INTEGER tmvar, istep, grid, taxis

* locate the step file that belongs to this data set
      DO 100 istep = 1,maxstepfiles
         IF (sf_setnum(istep) .EQ. dset) GOTO 190
 100  CONTINUE
* No step file allocated
      RETURN    ! can never happen
      
* locate the first variable with a valid time axis.  This will be 
* regarded as THE time axis for the data set
 190  DO 200 tmvar = 1,maxvars
        IF ( ds_var_setnum(tmvar) .NE. dset ) GOTO 200
        grid = ds_grid_number( tmvar )
        taxis = grid_line(t_dim, grid)
        IF ( taxis .NE. mpsnorm ) GOTO 300
 200  CONTINUE

* no variables with time axes in the data set
      ds_time_axis(dset) = mpsnorm 
      sf_1step(istep)    = unspecified_val8
      sf_lastep(istep)   = unspecified_val8
      GOTO 1000

* save the time axis and its limits
* add check for MC datasets....their sf_1step and sf_lastep are already
* set...dont want to clobber them
 300  IF (ds_type(dset) .NE. '  MC') then
         ds_time_axis(dset) = taxis
         sf_1step(istep)    = TM_WORLD( ds_grid_start(t_dim,tmvar),
     .                               grid, recdim, box_lo_lim )
         sf_lastep(istep)   = TM_WORLD( ds_grid_end(t_dim,tmvar),
     .                               grid, recdim, box_hi_lim )
      ENDIF

* successful completion
 1000   status = merr_ok
        RETURN

* error(s)
* 5000   RETURN
* 5100   CALL TM_ERRMSG ( merr_varform, status, 'CD_GET_TIME_AXIS',
*     .       unspecified_int4, unspecified_int4,
*     .       'no variables in netCDF file', no_errstring, *5000 )
	END
