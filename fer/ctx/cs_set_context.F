	SUBROUTINE CS_SET_CONTEXT

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
* transfer the information indicated in the current control stack level into
* the default ("LAST") context

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system

* revision 0.0 - 5/1/87
* V200:  6/15/89 - reordered lo/hi_ss args
* 9/97 *kob* - need to reset cx_calendar to false if cx_lo/hi_ww are
*              not used
* 12/97 *kob* - and set  cx_calendar to true if they are used....
* internal variable declarations:
	INTEGER	idim

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/xcontext.cmn'
	include 'ferret_cmn/xcontrol.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE 'FERRET_CMN:XCONTROL.CMN'
#endif

* initialize
	idim = cs_axis( csp )

* set up by subscript or by world coordinates
	cx_by_ss( idim, cx_last ) = cs_by_ss( csp )

	IF ( cs_by_ss( csp ) ) THEN
	   cx_lo_ss( cx_last, idim ) = cs_phase( csp )
	   cx_hi_ss( cx_last, idim ) = cs_phase( csp )
	   cx_lo_ww( idim, cx_last ) = unspecified_val8
	   cx_hi_ww( idim, cx_last ) = unspecified_val8
* kob 9/97 - need to reset cx_calendar as well
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx_last) = .FALSE.
	ELSE
* kob 12/97 - i need to set cx_calendar to true here
 	   cx_lo_ss( cx_last, idim ) = unspecified_int4
	   cx_hi_ss( cx_last, idim ) = unspecified_int4
	   cx_lo_ww( idim, cx_last ) = cs_phase( csp )
	   cx_hi_ww( idim, cx_last ) = cs_phase( csp )
	   IF (idim .eq. t_dim) 
     .                cx_calendar(cx_last) = cs_is_calendar(csp)
	ENDIF
	   
	RETURN
	END

