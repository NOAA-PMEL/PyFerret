	INTEGER FUNCTION TM_AXES_REVERSED( var, axes )

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
* return an array of length 4 indicating containing the axes that are reversed
* and the value of the function is the number that are reversed

* ... to hide TMAP COMMON from FERRET

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V320: 2/16/94

#ifdef unix
	include 'tmap_format/tmap_dims.parm'
#include "tmap_format/tmap_dset.parm"
	include 'tmap_format/xdset_info.cmn_text'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN_TEXT'
#endif

* calling argument declarations:
* *kob* 8/95 - changed decl. of axes from LOGICAL to INTEGER
	INTEGER	axes(4)
	INTEGER	var

* internal variable declarations
	INTEGER	i, number

* initialze
	number = 0

* check all the axes
* note that for netCDF files the array ds_ordering contains a list of only
* those dimensions that are relevant to the variable.  For example, the
* variable SST, which contains axes X (1), Y (2), and T (3) would have
* ds_ordering(1-4,SST_var) = 1,2,4,0
	DO 100 i = 1, 4
	   IF ( ds_ordering(i,var) .LT. 0 ) THEN
	      number = number + 1
	      axes(number) = ABS(ds_ordering(i,var))
	   ENDIF
 100	CONTINUE	   

	TM_AXES_REVERSED = number

	RETURN
	END
