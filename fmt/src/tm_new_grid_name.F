	SUBROUTINE TM_NEW_GRID_NAME ( suggested_name, new_name )
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
* If necessary creates a unique grid_name based on the suggested name by
* appending a number to the end of the suggested name

* Warning: if new_name is less than 5 characters this routine may STOP
*
* revision 0.00 - 11/22/88
*
* Arguments
	CHARACTER*(*)	suggested_name, new_name
*                                           
* Parameters and commons                    
#ifdef unix
        include 'tmap_dims.parm'
        include 'xtm_grid.cmn_text'  
	external xgt_grid_data
#else
        INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
        INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'  
#endif
*
* Local declarations
	REAL		newnum
	INTEGER*4	suglen, outlen, addlen, uselen, igrd 
	INTEGER*4	TM_LENSTR1
	CHARACTER*4	TM_FMT, add_on
*
* initialize
	suglen   = TM_LENSTR1(suggested_name)
	outlen   = LEN( new_name )
	new_name = suggested_name
	newnum   = 0.

* check for an identical name
  10	DO 100 igrd = 1, max_grids
	  IF (grid_name(igrd) .EQ. new_name) GOTO 200
  100	CONTINUE

* clean bill of health - no name match
	RETURN
*
* yes there's a duplicate - try to "uniqueify" the name
  200	newnum = newnum + 1.
	add_on = TM_FMT(newnum, 4, 4, addlen )
	uselen  = MIN( suglen, outlen-addlen )
	IF ( uselen .LE. 0 ) STOP 'TM_NEW_GRID_NAME'
	new_name = suggested_name(:uselen)//add_on
*
* no danger of an infinite loop unless there are over 9999 names
	GOTO 10
	END
