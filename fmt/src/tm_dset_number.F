	INTEGER FUNCTION TM_DSET_NUMBER( path_name )
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
* Find the data set number of the given name
* If the name does not match any initialized data sets return unspecified_int4
*
* Programmer Steve Hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX/VMS
*
* revision 0.00 - 10/23/89 - code lifted from TM_INIT_DSET by Mark Verschell

* arguments:
*	path_name	- full path name of descriptor file

* Argument definitions
	CHARACTER*(*)	path_name

* Parameter and common files
#ifdef unix
	include 'tmap_format/tmap_errors.parm'		! error codes
	include 'tmap_format/tmap_dims.parm'		! data set dimensions
	include 'tmap_format/gt_lib.parm'
	include 'tmap_format/xdset_info.cmn_text'		! data set common
	external xdset_info_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'		! Error codes
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'		! Data set dimensions
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'		! Data set common
#endif

* Internal declarations
	INTEGER		set_cnt

* check to see if this descriptor has already been initialized
	DO 100 set_cnt = 1, maxdsets
	    IF (ds_des_name(set_cnt) .EQ. path_name) THEN
* ... found a match
	      TM_DSET_NUMBER = set_cnt
	      RETURN
	    ENDIF
  100	CONTINUE

* no match (but no serious error)
	TM_DSET_NUMBER = unspecified_int4
	RETURN
	END
