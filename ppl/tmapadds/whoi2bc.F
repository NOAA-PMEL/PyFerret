	REAL*8 FUNCTION WHOI2BC( whoi )

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
* Steve Hankin - NOAA/PMEL/TMAP
*
* version 0.0 - based on TM_SECS_FROM_BC by Mark Verschell
* version 0.1 - calls TM_SECS_FROM_BC
* version 0.2 - bug fix: add 1900 to year
* Mod 11.22.95 J Davison for 4 digit year 
*
C *kob* 04/07/96 passed in character strings had to be declared 
C                with "*(*)" for the linux port 

* Argument definition
	CHARACTER*(*)	whoi
*
* Local Definition
	INTEGER		year, month, day, hour, minute, second
	INTEGER		century
	REAL*8		TM_SECS_FROM_BC

* read the WHOI-formatted string
	READ ( whoi, 3000 ) year, month, day, hour, minute, second, century
 3000	FORMAT (7I2.2)
!!!!!	second = 0
	year = year + century*100

* call TMAP library routine and convert seconds to minutes
	WHOI2BC = TM_SECS_FROM_BC( year, month, day, hour, minute, second )
     .		  / 60.D0

	RETURN
	END
