	SUBROUTINE SHOW_DATA_FILES( lun, dset, sbrief )

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
* show the files and time step ranges for the requested data set

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 8/15/88
* V230: 6/92 - elaborated on kludge to locate time axis for data set
*            - use the first variable that HAS a time axis
*              (not clear why ds_time_axis cannot be used ??)
* V301: 4/94 - show file for T-independent data set, too
*	       but not for a netCDF data set
* V320:12/94 - increased the output field size for /COLUMNS=xxxx
*Linux Port 1/97 *kob* - Added a preprocessor include for tmap_dset.parm
*			  because it needed a preprocessor.
*           *acm* 3/12 cleanup ifdefs and unnecessary include files
* V680 5/12 *acm* 6D Ferret, changes for DEFINE DATA/AGGREGATE
* V685 *acm*  3/13 For Ensemble aggregations, call the ds_type 'ENS'
* V695 *acm*  2/15 For Forecast aggregations, call the ds_type 'FCT'
* V695 *sh*  12/15 For FMRCs and Ensembles show aggregation members
* V698 *sh*   2/16 insert blank line at conclusion of list
* V702 *sh*  11/16 number the member files with the syntax (e.g.) "12.3"
* V71 12/16 *acm* For ticket 2158: working with true monthly time axes.
* V73  4/17 *acm* Change the text when showing members of a union aggregation,
*                 implement dataset.member notation for unions, (e.g.) "12.3"
* V76  2/20 *acm* Cticket 1965: show data/members/brief just lists the members, not timesteps etc.
* 11/2022 *acm* Code cleanup: Modernize DO-loop syntax. 


        IMPLICIT NONE
	include 'tmap_dims.parm'
#	include "tmap_dset.parm"
	include 'xdset_info.cmn_text'
	external xdset_info_data
	include 'xstep_files.cmn_text'
	external xstep_files_data
	include 'xtm_grid.cmn_text'
	external xgt_grid_data
	include	'ferret.parm'
	include	'errmsg.parm'
	include	'xprog_state.cmn'

* calling argument declarations:
	LOGICAL	sbrief
	INTEGER	lun, dset
	
* local variable declarations:
	LOGICAL		true_month
	INTEGER		KNOWN_GRID, TM_LENSTR1,
     .			ivar, grid, istep, ndec, status, lskip, slen,
     .			alen, nmemb, memb_set, nfig
	CHARACTER	LEFINT*8, lo_date*20, hi_date*20, buff*8, abuff*8
	REAL		timefac

* special processing of EZ data sets
	IF ( ds_type(dset) .EQ. pds_type_ez ) THEN
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /FORMAT = '//ds_parm_text(1,dset), 0)
           DO istep = 1,maxstepfiles
 	      IF (sf_setnum(istep) .EQ. dset) GOTO 20
 	   ENDDO
	   CALL ERRMSG( ferr_internal, status, 'EZ_ err', *5000 )
 20	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /SKIP = '//LEFINT( sf_skip(istep), lskip ), 0)
	   CALL SPLIT_LIST(pttmode_explct, lun,
     .		' /COLUMNS = '//LEFINT( ds_ntuple(dset), lskip ), 0)
 5000	   RETURN

* show aggregation members
        ELSEIF (ds_type(dset) .EQ. 'ENS'
     .    .OR.  ds_type(dset) .EQ. 'FCT'
     .    .OR.  ds_type(dset) .EQ. 'UNI') THEN

* ... get parent aggregation dataset number in string form
	   abuff = LEFINT(dset, alen )

* ... get number of aggregation members
	   CALL CD_GET_AGG_DSET_INFO (dset, nmemb, status)
	   IF (status .NE. ferr_ok) RETURN   ! should never happen
	   IF (.NOT. sbrief) CALL SPLIT_LIST(pttmode_explct, lun,
     .		'          Member datasets:', 0 )

* ... loop through the aggregation members.
	   nfig = LOG10(REAL(nmemb)) + 1
	   DO istep = 1, nmemb
	      IF (sbrief) THEN
	         CALL SPLIT_LIST(pttmode_explct, lun,
     .		   ds_des_name(memb_set), 0 )
	      ELSE
	         CALL CD_GET_AGG_DSET_MEMBER (dset, istep, memb_set, status)
	         IF (status .NE. ferr_ok) RETURN   ! should never happen
	         buff = LEFINT(istep, slen )
	         CALL SPLIT_LIST(pttmode_explct, lun,
     .		'      '//abuff(:alen)//'.'//buff(:nfig)//': '
     .			       //ds_des_name(memb_set), 0 )
	      ENDIF
	   ENDDO
	   RETURN

* nothing to show for netCDF files (4/94)
        ELSEIF (ds_type(dset) .EQ. 'CDF'
     .    .OR.  ds_type(dset) .EQ. 'ECDF') THEN
	   RETURN
	ENDIF

* !kludge - get the time axis from a variable in this data set
	DO 100 ivar = 1, maxvars
	   IF ( ds_var_setnum(ivar) .EQ. dset ) THEN
	      grid = KNOWN_GRID( dset, cat_file_var, ivar )
              IF ( grid_line(t_dim,grid) .NE. mnormal ) GOTO 200
	   ENDIF
 100	CONTINUE

******* 4/94 addition ******
* a time-independent data set
* list files belonging to this data set (should be one, only) 
	DO 110 istep = 1, maxstepfiles
	    IF (sbrief) THEN
	       IF ( sf_setnum(istep) .EQ. dset ) CALL SPLIT_LIST
     .		(pttmode_explct, lun, sf_name(istep), 0 )
	    ELSE
	       IF ( sf_setnum(istep) .EQ. dset ) CALL SPLIT_LIST
     .		(pttmode_explct, lun, ' time-independent data file: '
     .		//sf_name(istep), 0 )
	    ENDIF
 110	CONTINUE
******* end of 4/94 addition ******
	RETURN

* a time-dependent data set:
* write heading line for step file list
 200	CONTINUE
	IF (.NOT.sbrief) CALL SPLIT_LIST(pttmode_explct, lun,
     .		'          time-dependent data files:', 36)

* output string precision (from SET MODE CALENDAR)
           ndec = ax_dec_pt( t_dim )

	   CALL TM_MONTH_TIME_FACTOR (grid, true_month, timefac)

* see which step files belong with this data set 
	DO 300 istep = 1, maxstepfiles
	    IF ( sf_setnum(istep) .EQ. dset ) THEN
	      
	      IF (sbrief) THEN
	      
	         CALL SPLIT_LIST(pttmode_explct, lun,
     .		   sf_name(istep), 0 )

	      ELSE 
	         CALL TRANSLATE_TO_WORLD(	timefac* sf_1step(istep), t_dim, grid,
     .					ndec, lo_date )
	         CALL TRANSLATE_TO_WORLD(	timefac* sf_lastep(istep), t_dim, grid,
     .					ndec, hi_date )
	         slen = TM_LENSTR1(hi_date)
	         CALL SPLIT_LIST(pttmode_explct, lun,
     .		   lo_date(:slen)//' -> '//hi_date(:slen)//'    '
     .		   //sf_name(istep), 0 )
	      ENDIF
	   ENDIF
 300	CONTINUE

* insert blank line for nicer formatting
	CALL SPLIT_LIST(pttmode_explct, lun, ' ', 1)

	RETURN
	END

