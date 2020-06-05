	LOGICAL FUNCTION ITS_FMRC_2DTIME (dset, fvar)

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
* Return boolean indicating whether the passed variable is in fact
* the 2D FMRC time matrix of the dataset

* V698 2/16 *sh*

      IMPLICIT NONE
#include "tmap_dset.parm"
!#include "gt_lib.parm"     ! str_match
        include 'tmap_dims.parm'
	include	'ferret.parm'
	include 'errmsg.parm'
	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'
	include 'xtm_grid.cmn_text'

* calling argument declarations:
	INTEGER dset, fvar

* local variable declarations:
	LOGICAL NC_GET_ATTRIB, got_it
	INTEGER STR_CASE_BLIND_COMPARE, grid, vid, istat, status

	INTEGER str_match
        PARAMETER (str_match = 0)

* Getting info back from the attribute-handling structure
        INTEGER maxlen
        PARAMETER (maxlen = 128)
	CHARACTER*(maxlen) buff
	INTEGER varid, t2did, tline, g2d, globalid, ref_id, attlen,
     .		attype, attoutflag
        REAL    attvals(10)

* initialize
	grid = ds_grid_number(fvar)
* just in case this gets called with an inappropriate variable
        IF ( grid .EQ. int4_init ) GOTO 2000

* it must be 2D in the TF plane
	IF (grid_line(x_dim,grid) .NE. mnormal
     . .OR. grid_line(y_dim,grid) .NE. mnormal
     . .OR. grid_line(z_dim,grid) .NE. mnormal
     . .OR. grid_line(e_dim,grid) .NE. mnormal ) GOTO 2000
	IF (grid_line(t_dim,grid) .EQ. mnormal
     . .OR. grid_line(f_dim,grid) .EQ. mnormal ) GOTO 2000

* it must have a _CoordinateAxisType='Time' attribute
* ... get the List id of the variable
        CALL CD_GET_VAR_ID (dset, ds_var_code(fvar), vid, status)
        IF (status .NE. ferr_ok) GOTO 5100
* ... get the attribute val
        got_it = NC_GET_ATTRIB ( dset, vid, '_CoordinateAxisType',
     .                              .TRUE., ds_var_code(fvar), maxlen,
     .                              attlen, attoutflag, buff, attvals )
	IF (.NOT.got_it) GOTO 2000
	istat = STR_CASE_BLIND_COMPARE(buff, 'time')

* final test
        ITS_FMRC_2DTIME = istat .EQ. str_match
	RETURN

* nope - this aint it
 2000	ITS_FMRC_2DTIME = .FALSE.
	RETURN

* error bail out
 5000	RETURN
 5100	CALL ERRMSG(ferr_internal, status, 'its_fmrc_2dtime!',
     .		 *5000)     ! error not trapped
	END
