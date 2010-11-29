	CHARACTER*(*) FUNCTION GNAME_FROM_NUMBER( dset, grid, len )

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
* given a grid number return the name of the first file variable in this dset
* that is defined on this grid.  If the variable is from a different
* data set than the one passed in the call qualify the variable name with
* [D=name]
* If no variables are defined on this grid then return the name of the grid
* itself.

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/1/88
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.

#ifdef unix
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include	'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE	'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
#endif

* calling argument declarations:
	INTEGER		dset, grid, len

* local variable declarations:
	LOGICAL	in_set
	INTEGER	TM_LENSTR1, var, dset2, slen

* search for a file variable on both the right grid and data set
	DO 100 var = 1, maxvars
	   IF ( ds_var_setnum (var) .EQ. set_not_open ) GOTO 100

	   IF ( ds_grid_number(var) .EQ. grid
     .	  .AND. ds_var_setnum (var) .EQ. dset ) GOTO 400

 100	CONTINUE

* no luck - search for a file variable on the right grid in another data set
	DO 200 var = 1, maxvars
	   IF ( ds_var_setnum (var) .EQ. set_not_open ) GOTO 200

	   IF ( ds_grid_number(var) .EQ. grid ) GOTO 300

 200	CONTINUE

		         
* no luck again - no file variables are defined on this grid - return grid name
	GNAME_FROM_NUMBER = grid_name( grid )
	GOTO 500

* right grid but another data set
 300	dset2 = ds_var_setnum(var)
        CALL string_array_get_strlen1(ds_var_code_head, var, slen)
	GNAME_FROM_NUMBER = 
     .           ds_var_code(var)(:slen)//
     .		'[D='//ds_name(dset2)(:TM_LENSTR1(ds_name(dset2)))//']'
	GOTO 500

* right grid and variable
 400	GNAME_FROM_NUMBER = ds_var_code(var)

* measure the string	   
 500	len = TM_LENSTR1( GNAME_FROM_NUMBER )
	RETURN
	END
