	SUBROUTINE SET_VAR_ATT_VAL(dset, var, attname, vals, nvals, outflag)

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
* Convenience routine to easily create a new numerical attribute,
* <attname>, on variable ivar with values given in the calling argument.
* This routine should be used only when the attribute is 
* known not to exist yet.

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 5/17 *sh*

	include 'netcdf.inc'
	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdset_info.cd_equiv'

#include "tmap_dset.parm"

* calling arguments
*	dset	(input) dataset number in XDSET_INFO
*	var	(input) var number in XDSET_INFO
*	attname	(input) name of numerical attribute
*	vals	(input) value(s) found on attribute
*	nval	(input) number of values returned
*	outflag (input) 1 if attribute to be written to output files, 0 if not

* calling argument declarations:
	INTEGER dset, var, nvals, outflag
	CHARACTER*(*) attname
	REAL*8 vals(nvals)

* internal variable declarations:
        LOGICAL NC_GET_ATTRIB, got_it
        INTEGER vid, attoutflag, status
        CHARACTER    attbuff*10, vname*8

*** NOTE: THIS REMAINED UNTESTED CODE AT THE TIME OF ITS INITIAL WRITING
***  Is the datatype it created consistent with get_var_att_val ?


* initiailze
	attbuff = ' '     ! ??? what for ?

* get the name and id of the variable 
	vname = ds_var_code(var)  ! just for err msgs
	vid = cd_varid(var)

* attach the attribute
        CALL CD_PUT_NEW_ATTR_DP (dset, vid, attname, NCDOUBLE, nvalS,
     .               outflag, attbuff, vals, status)

	RETURN
	END
