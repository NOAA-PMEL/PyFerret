	SUBROUTINE SUB_2_GRIDS( plus, mplus, mnus, mmnus, rslt, mrslt )

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
* subtract 2 grids producing a third

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 6/25/87
* V200:  7/26/89 - 4D symmetrical version
*	10/12/89 - use ..._SUB - avoid adjustable dimensioning with array

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xvariables.cmn'
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XVARIABLES.CMN'
#endif

* calling argument declarations:
	INTEGER mplus, mmnus, mrslt
	REAL    plus( * ), mnus( * ), rslt( * )

* --- end of introductory code ---

	CALL SUB_2_GRIDS_SUB(
     .			  plus, mr_lo_s1(mplus), mr_hi_s1(mplus),
     .				mr_lo_s2(mplus), mr_hi_s2(mplus),
     .				mr_lo_s3(mplus), mr_hi_s3(mplus),
     .				mr_lo_s4(mplus), mr_hi_s4(mplus),
     .				mr_bad_data( mplus ),
     .			  mnus, mr_lo_s1(mmnus), mr_hi_s1(mmnus),
     .				mr_lo_s2(mmnus), mr_hi_s2(mmnus),
     .				mr_lo_s3(mmnus), mr_hi_s3(mmnus),
     .				mr_lo_s4(mmnus), mr_hi_s4(mmnus),
     .				mr_bad_data( mmnus ),
     .			  rslt, mr_lo_s1(mrslt), mr_hi_s1(mrslt),
     .				mr_lo_s2(mrslt), mr_hi_s2(mrslt),
     .				mr_lo_s3(mrslt), mr_hi_s3(mrslt),
     .				mr_lo_s4(mrslt), mr_hi_s4(mrslt),
     .				mr_bad_data( mrslt )			)


	RETURN
	END 
