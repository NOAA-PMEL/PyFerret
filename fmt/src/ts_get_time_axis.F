	INTEGER*4 FUNCTION TS_GET_TIME_AXIS ( dset_num, ivar, status )
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
*  Will use the information stored in XDSET_INFO.CMN to construct a temporary
*  time axis in XTM_GRID.CMN which will then be compared to pre-existing grids
*  stored in common. If a match is found the temporary axis is deleted and the
*  line number for the pre-existing axis is returned. If no match is found,
*  the temporary axis is moved to permanent storage, and it's line number is
*  returned.
*
*  Written 07/88 for PMEL/TMAP by M. Verschell
*
* revision 0.00 - 07/28/88
* revision 0.01 - 11/21/88 - variable used check changed to specific char_initXX
* bug fix - 10/2/91 - commented out the 9000 statement, as there was no path to it.
*           caught by ver 3 of f77, but not 2.1   kob
* Linux port -kob - 3/97 - Modify include for tmap_dset.parm to be a
*			   preprocessor include for F90 - needed because
*			   tmap_dset.parm contains an ifdef
* V5.3  *acm*  adding calendar types: cal_id

* CALLING ARGUMENTS
*	dset_num - the data set fow which a time axis is desired
*	ivar - variable axis is being made for
*	status  - the returning status
	INTEGER*4	dset_num, ivar, status
*
* INCLUDE FILES
#ifdef unix
	include 'tmap_dims.parm'
#include "tmap_dset.parm"
#include "gt_lib.parm"
	include 'tmap_errors.parm'
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include 'xunits.cmn_text'
	external xunits_data
	include 'xio.cmn_text'
	external xio_data
#else
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DSET.PARM'
	INCLUDE 'TMAP_FORMAT:GT_LIB.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'TMAP_FORMAT:XTM_GRID.CMN'
	INCLUDE 'TMAP_FORMAT:XUNITS.CMN'
	INCLUDE 'TMAP_FORMAT:XIO.CMN'
#endif
*
* LOCAL DEFINITIONS
	CHARACTER*13	TM_STRING
	INTEGER*4	scr, unused, nsteps, ilne, itime, ltime, iunit,
     .			i1, i2, i3, i4, i5, i6, old, nline
	INTEGER*4	TM_FIND_LIKE_LINE
        INTEGER TM_GET_CALENDAR_ID, cal_id


	PARAMETER	( scr = 0 )

* Get variable copy of parameter
	unused   = int4_init

* Fill in unneeded variable and "already known" quantities
	line_subsc1(scr)    = unspecified_int4
	line_regular(scr)   = .TRUE.
	line_modulo(scr)    = .FALSE.

* Fill in axis particulars
	line_start(scr)     = min_ds_tstart
	line_delta(scr)     = ds_tdelta

* Time_step units
	line_t0(scr)        = ds_t0time(dset_num)
	line_tunit(scr)     = ds_time_unit(dset_num)
	DO 100 iunit = pun_1st_time, pun_last_time
	  IF ( line_tunit(scr) .EQ. un_convert(iunit) ) THEN
	    line_units(scr)     = un_name(iunit)
	    line_unit_code(scr) = iunit
	    GOTO 110
	  ENDIF
  100	CONTINUE
	line_units(scr)     = ' '
	line_unit_code(scr) = 0

        line_cal_name(scr) = ds_cal_name(dset_num)

* Check if time/date string conversion is possible
  110	CONTINUE
        cal_id = TM_GET_CALENDAR_ID (line_cal_name(scr))
        CALL TM_BREAK_DATE( line_t0(scr), cal_id,
     .                      i1,i2,i3,i4,i5,i6,status )
	IF ( line_tunit(scr) .LE. 0       .OR.
     .	     line_t0(scr)    .EQ. ' '     .OR.
     .	     status          .NE. merr_ok      ) THEN
	  line_direction(scr) = 'NA'
	  WRITE (tooplu, 1000) ds_name(dset_num)
 1000	  FORMAT (' Warning: improper date/time format information in ',A)
	ELSE
	  line_direction(scr) = 'TI'
	ENDIF

* Determine the length of the time axis
	line_dim(scr) = 1 + (max_ds_tend-min_ds_tstart)/ds_tdelta

* Time axis created, fill in the name.
	CALL TM_NEW_LINE_NAME('TIME',line_name(scr))

* ...Compare this with existing axis
	CALL TM_GET_LIKE_LINE(scr,line_name(scr),nline,status)
	IF (status .NE. merr_ok) GOTO 9100

* Time axis complete
	TS_GET_TIME_AXIS = nline
	status = merr_ok
	GOTO 9999

* 9000	CALL TM_ERRMSG (merr_notimeinfo, status, 'TS_GET_TIME_AXIS',
*     .			dset_num, no_stepfile,
*     .			no_errstring, no_errstring,*9999)

 9100	CALL TM_ERRMSG (merr_linelim, status, 'TS_GET_TIME_AXIS',
     .			dset_num, no_stepfile,
     .			'MAX='//TM_STRING(DBLE(max_lines)), no_errstring,
     .			*9999)

* Exit subroutine
 9999	RETURN
	END
