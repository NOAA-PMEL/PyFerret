	SUBROUTINE PPL_AXES_RESTORE

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
* restore the PLOT+ axes after they have been turned off by PPL_AXES_OFF

* V450:  

#ifdef unix
	include 'ferret_cmn/plot_setup.parm'
	include 'tmap_pplv11inc/axis_inc.decl'
	include 'pplv11inc/AXIS.INC'		! variable iaxon
#else
	INCLUDE 'FERRET_CMN:PLOT_SETUP.PARM'
	INCLUDE 'TMAP_PPLV11INC:AXIS_INC.DECL'
	INCLUDE 'PPLV11INC:AXIS.INC'	
#endif

* calling argument declarations

* local variable declarations:
	INTEGER i
	CHARACTER buff*16

* static storage in COMMON
	INTEGER save_iaxon(4)
	COMMON /XAXIS_STATE/ save_iaxon

* convert 4 axis integers to character as arg to the AXSET command
	buff = ' '
	WRITE (buff, '(4I3)' ) (save_iaxon(i), i=1, 4)

* restore axis state
	CALL PPLCMD ( from, line, 0, 'AXSET '//buff, 1, 1 )

	RETURN
	END
