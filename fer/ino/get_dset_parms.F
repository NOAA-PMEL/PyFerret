	SUBROUTINE GET_DSET_PARMS( dset, status )

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
* get parameter values for data set - 1st set defaults, then read (if possible)
* from the 205 data base, then over-ride using any parameters given in the
* descriptor

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* revision 0.0 - 2/11/88
* revision 0.1 - 4/28/88 - get parameters from 205 data base
* revision 0.2 - 6/22/88 - allow SIEG and SEIG interchangeably
*			 - warning only for invalid 205 data base entry
* revision 0.3 - 7/1/88  - allow multiple air temp climatologies
*			 - add dp_Am_factor
* revision 1.0 -11/17/88 - new io library and grid common
* revision 1.1 - 4/19/89 - changed AIRT logic so that 3-letter code is read
*			   from 205 data base.  "LEV" is still default.
* Unix/RISC port 5/91 *sh*: changed runs data base file name
*               8/29/91 - added add'l parm "NTS30 WIND STRESS CALC"
* V320: 12/29/94 - use SPLIT_LIST to redirect tty output for GUI
* V420: 9/95 - using TM_ITSA_MODEL_RUN TMAP runs are now recognizable from
* 		their file names, alone.  Parameters are then read from the
*		205 run data base.

#ifdef unix
	include 'ferret_cmn/ferret.parm'
	include 'ferret_cmn/errmsg.parm'
	include 'tmap_format/tmap_dims.parm'
	include 'tmap_format/tmap_errors.parm'
	include 'ferret_cmn/xdset_parms.cmn'
	include 'ferret_cmn/xprog_state.cmn'
	include 'tmap_format/xdset_info.cmn_text'
	external xdset_info_data
	include 'ferret_cmn/xrisc.cmn'          ! 12/94 SPLIT_LIST buffer
#else
	INCLUDE 'FERRET_CMN:FERRET.PARM'
	INCLUDE 'FERRET_CMN:ERRMSG.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_DIMS.PARM'
	INCLUDE 'TMAP_FORMAT:TMAP_ERRORS.PARM'
	INCLUDE 'FERRET_CMN:XDSET_PARMS.CMN'
	INCLUDE 'FERRET_CMN:XPROG_STATE.CMN'
	INCLUDE 'TMAP_FORMAT:XDSET_INFO.CMN'
	INCLUDE 'FERRET_CMN:XRISC.CMN'          ! 12/94 SPLIT_LIST BUFFER
#endif

* calling argument declarations
	INTEGER	dset, num_prev_gridfiles, status

* local variable declarations
	LOGICAL	TM_HAS_STRING, TM_DIGIT, TM_ITSA_MODEL_RUN
	INTEGER	TM_LENSTR1,
     .		NEXT205, TMAP_status, dummy, i, slen, iclim
	CHARACTER	TM_EXP_CODE*6, col_head*132, table_row*132,
     .			entry*64, entry_nam*64

* parameter definitions
	INTEGER	    max_dset_parms
	PARAMETER ( max_dset_parms = 15 )	! to match *DSET_INFO.CMN

* Philander/Siegel model output ? (Sieg or Seig)
	dp_phil_sieg( dset ) = TM_HAS_STRING( ds_source(dset), 'PHIL' )
     .			 .AND. ( TM_HAS_STRING( ds_source(dset), 'SIEG' )
     .			    .OR. TM_HAS_STRING( ds_source(dset), 'SEIG' ) )

* 9/95 - may conclude it is a TMAP model run based on file name, too
* (the term "Philander/Siegel" has been commandeered for all TMAp runs)
	IF (.NOT.dp_phil_sieg(dset)) dp_phil_sieg(dset) =TM_ITSA_MODEL_RUN
     .					(ds_name(dset),
     .					 ds_source(dset),
     .					 ds_expnum(dset)  )

* Cox model output ?
	dp_cox( dset )	 = TM_HAS_STRING( ds_source(dset), 'COX' )

* something unknown ?
	IF (   ds_source(dset) .NE. ' '
     .	.AND. .NOT.dp_phil_sieg(dset)
     .	.AND. .NOT.dp_cox(dset) ) THEN
	   WRITE ( risc_buff, 3020 )
     .		ds_source(dset)( :TM_LENSTR1( ds_source(dset) ) ),
     .		ds_name(dset)( :TM_LENSTR1( ds_name(dset) ) )
	   CALL WARN ( risc_buff )
	ENDIF

* tropical Pacific (160x100x27) grid ?
	dp_trop_pacific( dset ) = TM_HAS_STRING( ds_subsource(dset), 'TROP' )
     .			    .AND. TM_HAS_STRING( ds_subsource(dset), 'PAC'  )

********* PHILANDER/COX parameters ****************************
* * * * INITIALIZE PARAMETER DEFAULTS * * * * 
	dp_min_wind	( dset ) = 488.
	dp_phil_qflux	( dset ) = .TRUE.
	dp_ramp1_qflux	( dset ) = .FALSE.
	dp_ramp2_qflux	( dset ) = .FALSE.
	dp_clim_air_sst	( dset ) = 1		! LEVITUS
	dp_const_air_sst( dset ) = .TRUE.
	dp_delta_air_sst( dset ) = 1.0
	dp_Am_factor	( dset ) = 2.0
        dp_nts30_wind   ( dset ) = .FALSE.

* * * *  CHECK THE 205 DATA BASE * * * * 
	IF ( dp_phil_sieg(dset).OR.dp_cox(dset) ) THEN
* get the table heading line and row containing this experiment code
	   CALL TM_GET_205_RUN_LINE
     .		( 'model_runs.doc', TM_EXP_CODE(dset), col_head, table_row,
     .			TMAP_status )	
	   IF ( TMAP_status .NE. merr_ok ) THEN
	      CALL WARN ('Model run parameters not available')
	      CALL WARN ('Diagnostic variable results may be incorrect')
	      GOTO 500
	   ENDIF

* ... MINIMUM WIND
	   ASSIGN 110 TO NEXT205
	   entry_nam = 'WND'
	   CALL TM_GET_205_RUN_PARM
     .		( entry_nam, col_head, table_row, entry, slen, TMAP_status )
	   IF ( TMAP_status .NE. merr_ok ) GOTO 5010
	   IF ( .NOT.TM_DIGIT(entry) ) GOTO 5010
	   READ ( entry, *, ERR=5010 ) dp_min_wind(dset)

* ... HEAT FLUX PARAMETERIZATION
 110	   ASSIGN 120 TO NEXT205
	   entry_nam = 'FLUX'
	   CALL TM_GET_205_RUN_PARM
     .		( entry_nam, col_head, table_row, entry, slen, TMAP_status )
	   IF ( TMAP_status .NE. merr_ok ) GOTO 5010
	   dp_phil_qflux  ( dset ) = TM_HAS_STRING( entry, 'PHIL' )
	   dp_ramp1_qflux ( dset ) = TM_HAS_STRING( entry, 'RAMP1' )
	   dp_ramp2_qflux ( dset ) = TM_HAS_STRING( entry, 'RAMP2' )
	   IF ( .NOT.dp_phil_qflux( dset )
     .	  .AND. .NOT.dp_ramp1_qflux( dset )
     .	  .AND. .NOT.dp_ramp2_qflux( dset ) ) GOTO 5010

* ... AIR/SST DELTA TEMP (a value or "CLIM")
 120	   ASSIGN 130 TO NEXT205
	   entry_nam = 'SEAT'
	   CALL TM_GET_205_RUN_PARM
     .		( entry_nam, col_head, table_row, entry, slen, TMAP_status )
	   IF ( TMAP_status .NE. merr_ok ) GOTO 5010
	   IF ( TM_DIGIT( entry ) ) THEN
	      READ ( entry, '(F24.0)', ERR=5010 ) dp_delta_air_sst(dset)
	      dp_delta_air_sst( dset ) = -dp_delta_air_sst(dset) ! neg. stored
	      dp_const_air_sst( dset ) = .TRUE.
	   ELSE
	      IF ( entry(:4) .NE. 'CLIM' ) GOTO 5010
	      dp_const_air_sst( dset ) = .FALSE.
	   ENDIF

* ... CLIMATOLOGICAL AIR TEMP FIELD
 130	   ASSIGN 140 TO NEXT205
	   entry_nam = 'AIR'
	   CALL TM_GET_205_RUN_PARM
     .		( entry_nam, col_head, table_row, entry, slen, TMAP_status )
	   IF ( TMAP_status .NE. merr_ok ) GOTO 5010
	   DO 135 iclim = 1, max_airt_clim
	      IF ( dp_airt_name(iclim) .EQ. unspecified_name4 ) GOTO 137
	      IF ( TM_HAS_STRING(entry,dp_airt_name(iclim)) ) THEN
* ... this 3-letter AIRT code is already known: point data set to it
	         dp_clim_air_sst ( dset ) = iclim
	         GOTO 140
	      ENDIF
 135	   CONTINUE
* ... a new 3-letter code but no room to store it
	   GOTO 5030
* ... a new 3-letter code: save it and point data set to it
 137	   dp_airt_name( iclim )   = entry
	   dp_clim_air_sst( dset ) = iclim

* ... Am FACTOR - HORIZONTAL MOMENTUM DIFFUSION
 140	   ASSIGN 500 TO NEXT205
	   entry_nam = 'Am'
	   CALL TM_GET_205_RUN_PARM
     .		( entry_nam, col_head, table_row, entry, slen, TMAP_status )
	   IF ( TMAP_status .NE. merr_ok ) GOTO 5010
	   IF ( .NOT.TM_DIGIT( entry ) ) GOTO 5010
	   READ ( entry, '(F24.0)', ERR=5010 ) dp_Am_factor(dset)
	ENDIF

* * * * DESCRIPTOR PARAMETERS * * * *
* "name" = "value"
 500	DO 590 i = 1, max_dset_parms

	   entry_nam = ds_parm_name(i,dset)
	   IF ( entry_nam .EQ. ' ' ) THEN
	      GOTO 590
	   ELSE
	      entry = ds_parm_text(i,dset)
	   ENDIF

* ... "HEAT FLUX = PHILANDER/SIEGEL" or "HEAT FLUX = DOUBLE RAMP", etc.
	   IF ( TM_HAS_STRING( entry_nam, 'HEAT FLUX'	) )THEN
	      dp_phil_qflux  ( dset ) = TM_HAS_STRING( entry, 'PHIL' )
	      dp_ramp1_qflux ( dset ) = TM_HAS_STRING( entry, 'SINGLE' )
	      dp_ramp2_qflux ( dset ) = TM_HAS_STRING( entry, 'DOUBLE' )

* ... "AIR-SEA DELTA T = value,"LEV","CAC",... "?
	   ELSEIF ( TM_HAS_STRING( entry_nam, 'AIR-SEA DELTA T'	) ) THEN
	   IF ( TM_DIGIT( entry ) ) THEN
	      READ ( entry, '(F24.0)', ERR=5010 ) dp_delta_air_sst(dset)
	      dp_delta_air_sst( dset ) = -dp_delta_air_sst(dset) ! neg. stored
	      dp_const_air_sst( dset ) = .TRUE.
	      dp_clim_air_sst ( dset ) = unspecified_int4
	   ELSE
	      DO 510 iclim = 1, max_airt_clim
	         IF ( TM_HAS_STRING(entry,dp_airt_name(iclim)) ) THEN
	            dp_const_air_sst( dset ) = .FALSE.
	            dp_clim_air_sst ( dset ) = iclim
	            GOTO 590
	         ENDIF
 510	      CONTINUE
	      GOTO 5020
	   ENDIF

* ... "MINIMUM WIND = 488."
	   ELSEIF ( TM_HAS_STRING( entry_nam, 'MINIMUM WIND' ) ) THEN
	      IF ( entry .EQ. ' ' ) GOTO 5020
	      IF ( TM_DIGIT( entry ) ) THEN
	         READ ( entry, *, ERR=5020 ) dp_min_wind(dset)
	      ELSE
	         GOTO 5020
	      ENDIF

* ... "Am_factor = 2.0"
	   ELSEIF ( TM_HAS_STRING( entry_nam, 'Am_factor' ) ) THEN
	      IF ( entry .EQ. ' ' ) GOTO 5020
	      IF ( TM_DIGIT( entry ) ) THEN
	         READ ( entry, *, ERR=5020 ) dp_Am_factor(dset)
	         IF ( dp_Am_factor(dset) .LT. 0.0 ) GOTO 5020
	      ELSE
	         GOTO 5020
	      ENDIF

* ... "205_AIRT = "LEV","CAC", "ALV", etc.
*     (flag this data set as containing climatological air temp ?)
	   ELSEIF ( TM_HAS_STRING( entry_nam, '205_AIRT' ) ) THEN
	      DO 550 iclim = 1, max_airt_clim
	         IF ( dp_airt_name(iclim) .EQ. unspecified_name4 ) THEN
*	... fill blank AIRT slot with this AIRT 3-letter code
	            dp_airt_name(iclim) = entry(:3)
	            dp_airt_dset(iclim) = dset
	            GOTO 590
	         ELSEIF (TM_HAS_STRING(entry,dp_airt_name(iclim)) ) THEN
*	... point this data set at the already specified 3-letter AIRT code
	            IF ( dp_airt_dset(iclim) .EQ. unspecified_int4 ) THEN
	               dp_airt_dset(iclim) = dset
	               GOTO 590
	            ELSE
	              GOTO 5019
	            ENDIF
	         ENDIF
 550	      CONTINUE
*	... no blank AIRT slots for this AIRT set
	      GOTO 5030

* ... "NTS30 WIND STRESS MODIFICATION"
	   ELSEIF ( TM_HAS_STRING( entry_nam, 'NTS30 WIND' ) ) THEN
              dp_nts30_wind( dset ) = .TRUE.

* ... parameter not recognized
	   ELSEIF ( dp_phil_sieg(dset).OR.dp_cox(dset) ) THEN
	      WRITE ( risc_buff, 3020 )
     .		entry_nam( :TM_LENSTR1( entry_nam ) ),
     .		ds_name(dset)( :TM_LENSTR1( ds_name(dset) ) )
 3020	      FORMAT( ' Warning: unrecognized parameter ', A,
     .					  ' in data set ', A )
	      CALL WARN ( risc_buff )
	   ENDIF

 590	CONTINUE

* successful completion
	status = ferr_ok
	RETURN

* syntax error
 5000	RETURN

 5010	CALL WARN('Invalid entry in model run data base')
	CALL WARN('Data set: '//ds_name(dset)(:TM_LENSTR1(ds_name(dset))))
	CALL WARN('Column:'//entry_nam(:TM_LENSTR1(entry_nam)))
	CALL WARN('Field:'//entry(:slen))
	CALL WARN( 'Continuing with errors ...')
	GOTO NEXT205
 5019	CALL WARN( 'MULTIPLE DSETS CLAIM TO BE SAME AIR TEMPERATURE SET')
 5020	CALL ERRMSG( ferr_descriptor, status,
     .		ds_name(dset)( :TM_LENSTR1( ds_name(dset) ) )//' : "'//
     .		entry_nam(:TM_LENSTR1(entry_nam))//'='//
     .		entry(:TM_LENSTR1(entry))//'"',
     .		*5000 )
 5030	CALL ERRMSG( ferr_prog_limit, status,
     .		ds_name(dset)( :TM_LENSTR1( ds_name(dset) ) )//
     .		' :no room for air temp data set '// entry(:4),
     .		*5000 )

	END
