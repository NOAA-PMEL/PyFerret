	DOUBLE PRECISION FUNCTION TDEST_CONVERT( tstep )

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
* convert a time step value to a new time encoding using slope and offset
* that were set up by TDEST_WORLD_INIT
* This routine exists to optimize performance in converting between time axes

* The conversion process is simply based on the fact that each time axis is
* a slope (units) and an intercept (T0)
* Let
*	A - absolute time (seconds since BC)
*	U - units (slope)
*	T - axis-relative time
* so that
*	AT0src - is the absolute time of T0 on the source axis
*	AT0dst - equivalently on the destination axis
* Then
*	Tsrc = (A-AT0src) / Usrc
*	Tdst = (A-AT0dst) / Udst
* from which
*	Tdst = Tsrc*(Usrc/Udst) + (AT0src-AT0dst)/Udst 
* or in terms of the COMMON variables
*	Tdst = Tscr*tcnvrt_slope + tcnvrt_intercept

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720: *sh* 5/17 - the code here lifted untouched from TDEST_WORLD

	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xregrid.cmn'

* calling argument declarations:
	REAL*8 tstep

* internal variable declarations:
        INTEGER status

* special treatment of calendar time axis
        IF ( tcnvrt_ok  ) THEN
	   TDEST_CONVERT = tstep * tcnvrt_slope + tcnvrt_intercept
	ELSE
	   CALL WARN('Time axis conversion not properly initialized')
	   TDEST_CONVERT = unspecified_val8  ! untested - likely to cause crash
	ENDIF

 5000	RETURN
	END
