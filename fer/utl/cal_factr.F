	REAL FUNCTION CAL_FACTR (cal_id)

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
* Find factor to adjust time units for different calendar axes.

* programmer - Ansley Manke
*
* revision 0.00 - 03-Mar-2001

* calling argument declarations:
	INTEGER cal_id

* local variable declarations
	INTEGER cal_id_greg
	REAL sec_per_year(6)

#ifdef unix
	include 'tmap_dims.parm'
	include 'xunits.cmn_text'
#endif

	DATA sec_per_year/31556952., 31536000., 31557600., 31104000.,
     .                    31104000., 0./

	cal_id_greg = 1		! Default gregorian calendar
	CAL_FACTR = sec_per_year(cal_id_greg) / sec_per_year(cal_id)


c from xunits_data.F:
c	DATA	un_convert( -1)	/ 1.0		/,
c     .		un_convert( -2)	/ 60.		/,
c     .		un_convert( -3)	/ 3600.		/, ! sec per hour
c     .		un_convert( -4)	/ 86400.	/, ! sec per day
c     .		un_convert( -5)	/ 2629746.	/, ! 1/12 of 365.2425 year cal 1
c     .		un_convert( -6)	/ 31536000.	/, ! sec per 365 day year  cal 2
c     .		un_convert( -7)	/ 44712.	/, ! sec per M2 cycle
c     .		un_convert( -8)	/ 31622400.	/, ! sec per 366 day year
c     .		un_convert( -9)	/ 31104000.	/, ! sec per 360 day year  cal 4,5
c     .		un_convert(-10)	/ 31556952.	/, ! sec per 365.2425 year cal 1
c     .		un_convert(-11)	/ 31557600.	/, ! sec per 365.25 day yr cal 3
c     .          un_convert(-12) / 2628000.	/, ! 1/12 of 365 day year  cal 2	
c     .          un_convert(-13) / 2635200.	/, ! 1/12 of 366 day year	
c     .          un_convert(-14) / 2592000.	/, ! 1/12 of 360 day year  cal 4,5
c     .		un_convert(-15)	/ 2629800.	/  ! 1/12 of 365.25 day yr cal 3

	RETURN
	END
