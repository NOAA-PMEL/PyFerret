	SUBROUTINE DEFINE_CLIMAT_ATT (src_cx)

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
* Ansley Manke March 2016
* V698 *acm*  3/16 Ticket 2278: add an attribute with time range when
*                  saving a modulo-regridded variable to netCDF

        IMPLICIT NONE
        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'xcontext.cmn'
	include 'xtra_compute_attr.cmn'


* calling argument declarations:
	INTEGER src_cx

* internal variable declarations:
	LOGICAL ssame
        INTEGER TM_LENSTR1, STR_SAME, grid, loss, hiss, slen, iprec
	CHARACTER*128 str1, str2
	REAL*8 TM_WORLD, loww, hiww, wwnext

	grid = cx_grid(src_cx)
	loss = cx_lo_ss(src_cx, t_dim)
	hiss = cx_hi_ss(src_cx, t_dim)

* Show month-year using the grid midpoints of the date range

	loww = TM_WORLD (loss, grid, t_dim, box_middle)
	hiww = TM_WORLD (hiss, grid, t_dim, box_middle)

	iprec = 2

	CALL TSTEP_TO_DATE ( grid, t_dim, loww, iprec, str1 )
      
	slen = TM_LENSTR1(str1)
	CALL TSTEP_TO_DATE ( grid, t_dim, hiww, iprec, str2 )

	str1 = str1(:slen)//':'//str2
	

* First check that this isn't already set.  If it is, and if the 
* new value of the string is different, we will not set the attrib.

	IF (natt_computed_var.GT.0) THEN  
	   slen = STR_SAME (attstr_computed_var, str1)
	   IF (STR_SAME (attname_computed_var, 'climatology_time_range') .NE. 0) 
     .                attstr_computed_var = 'do not set. another extra attribute already defined'
	   IF ( STR_SAME (attstr_computed_var, str1) .NE. 0) 
     .                attstr_computed_var = 'Multiple time ranges used'
	ELSE
	   natt_computed_var = 1
	   attname_computed_var = 'climatology_time_range'
	   attstr_computed_var = str1
	ENDIF

	RETURN
	END
