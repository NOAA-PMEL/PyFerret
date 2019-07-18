	SUBROUTINE DSG_COORD_LIMS(dset, idim, lolim, hilim)

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
* Return the lo/hi coordinate limits of the given axis of the dsg dataset

* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

* V720 4/17 *sh*
* V74+ 6/18 *acm* get actual_range for rowsize variable

	include 'tmap_dims.parm'
	include 'ferret.parm'
	include 'errmsg.parm'
	include 'xtm_grid.cmn_text'
	include 'xdset_info.cmn_text'
	include 'xdyn_linemem.cmn_text'
	include 'xcontext.cmn'
	include 'xprog_state.cmn'
	include 'xvariables.cmn'
#include "tmap_dset.parm"

* calling argument declarations:
	INTEGER dset, idim
	REAL lolim, hilim

* internal variable declarations:
        INTEGER TM_DSG_E_AXIS, var, nvals, line
	REAL	vals(2)

* local parameter definitions:
        INTEGER     maxlen
	PARAMETER ( maxlen = 2 )

* E axis
	IF (idim .EQ. e_dim) THEN
	   line = TM_DSG_E_AXIS(dset)
	   lolim = 1
	   hilim = line_dim(line)
	   RETURN
	ENDIF

* all other axes
	IF (idim .EQ. pdsg_row_size) THEN
	   var = dsg_row_size_var(dset)	
	ELSE
* ... find the DSG dataset variable associated with this coordinate
	   var = dsg_coord_var(idim,dset)
	ENDIF

	IF (var .NE. int4_init) THEN

* ... get the values stored on attribute 'actual_range'
*     these have reliably been set by CD_GET_DSG_COORD_EXTREMES
	   CALL GET_FVAR_ATT_VAL(dset, var, 'actual_range',
     .				   maxlen, vals, nvals)
	   lolim = vals(1)
	   hilim = vals(2)

	ELSE
* ... no limits exist
	   lolim = unspecified_val8
	   hilim = unspecified_val8
	ENDIF

	RETURN
	END
