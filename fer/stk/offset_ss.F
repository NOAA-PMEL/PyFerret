	SUBROUTINE OFFSET_SS( idim, cx, lo_off, hi_off, status )

* determine offsets to subscripts of the specified axis needed to request
* a region of component data adequate to compute the result region

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  5/31/89 - based on CHG_PLANE_LIMS
* for Unix/RISC port 9/91 - added SHN,SPZ,SWL
* V230:  1/22/92 - added @WEQ and @RSUM
*        3/23/92 - added @FLN, @FNR
*         7/2/92 - changed @FLN, @FNR to off-edge (vs. window width)  
* V455:  9/97 - trap illegal offset arguments, return status
*	      - added transforms @CDA,CDB,CIA,CIB

* calling argument declarations:
	INTEGER idim, cx, lo_off, hi_off, status

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/errmsg.parm'
	include	'ferret_cmn/xcontext.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:ERRMSG.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
#endif

* internal variable declarations:
	LOGICAL	offset_supplied( 2:max_plane_transform )
	INTEGER	ioff, trans, lout,
     .		offset_lo   ( 2:max_plane_transform ),
     .		offset_hi   ( 2:max_plane_transform )
	REAL	arg
	CHARACTER*64 VAR_TRANS, buff

* pre-stored offsets needed for transformations
	DATA offset_lo  	( trans_integ_indef )	/  0  /,
     .	     offset_hi  	( trans_integ_indef )	/  0  /,
     .	     offset_supplied	( trans_integ_indef )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_smth_box )	/ -1  /,
     .	     offset_hi   	( trans_smth_box )	/ +1  /,
     .	     offset_supplied	( trans_smth_box )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_deriv_cntr )	/ -1  /,
     .	     offset_hi   	( trans_deriv_cntr )	/ +1  /,
     .	     offset_supplied	( trans_deriv_cntr )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_deriv_fwrd )	/  0  /,
     .	     offset_hi   	( trans_deriv_fwrd )	/ +1  /,
     .	     offset_supplied	( trans_deriv_fwrd )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_deriv_bkwd )	/ -1  /,
     .	     offset_hi   	( trans_deriv_bkwd )	/  0  /,
     .	     offset_supplied	( trans_deriv_bkwd )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_subset )	/  0  /,
     .	     offset_hi   	( trans_subset )	/  0  /,
     .	     offset_supplied	( trans_subset )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_shift )		/ 1 /,
     .	     offset_hi   	( trans_shift )		/ 1 /,
     .	     offset_supplied	( trans_shift )		/ .TRUE. /,
     .
     .	     offset_lo   	( trans_fill_ave )	/ -1  /,
     .	     offset_hi   	( trans_fill_ave )	/ +1  /,
     .	     offset_supplied	( trans_fill_ave )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_smth_binml )	/ -1  /,
     .	     offset_hi   	( trans_smth_binml )	/ +1  /,
     .	     offset_supplied	( trans_smth_binml )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_smth_hanng )	/ -1  /,
     .	     offset_hi   	( trans_smth_hanng )	/ +1  /,
     .	     offset_supplied	( trans_smth_hanng )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_smth_parzn )	/ -1  /,
     .	     offset_hi   	( trans_smth_parzn )	/ +1  /,
     .	     offset_supplied	( trans_smth_parzn )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_smth_welch )	/ -1  /,
     .	     offset_hi   	( trans_smth_welch )	/ +1  /,
     .	     offset_supplied	( trans_smth_welch )	/ .TRUE. /,
     .
     .       offset_lo  	( trans_run_sum )	/  0  /,
     .	     offset_hi  	( trans_run_sum )	/  0  /,
     .	     offset_supplied	( trans_run_sum )	/ .FALSE. /,
     .
     .       offset_lo  	( trans_wgted_eq )	/  0  /,
     .	     offset_hi  	( trans_wgted_eq )	/  0  /,
     .	     offset_supplied	( trans_wgted_eq )	/ .FALSE. /,
     .
     .	     offset_lo   	( trans_fill_interp )	/ -1  /,
     .	     offset_hi   	( trans_fill_interp )	/ +1  /,
     .	     offset_supplied	( trans_fill_interp )	/ .TRUE. /,
     .
     .	     offset_lo   	( trans_fill_near )	/ -1  /,
     .	     offset_hi   	( trans_fill_near )	/ +1  /,
     .	     offset_supplied	( trans_fill_near )	/ .TRUE. /

	DATA offset_lo  	( trans_nrst_dist_abv )	/  -1 /,
     .	     offset_hi  	( trans_nrst_dist_abv )	/  +1 /,
     .	     offset_supplied	( trans_nrst_dist_abv )	/ .TRUE. /,
     .
     .	     offset_lo  	( trans_nrst_dist_blw )	/  -1 /,
     .	     offset_hi  	( trans_nrst_dist_blw )	/  +1 /,
     .	     offset_supplied	( trans_nrst_dist_blw )	/ .TRUE. /,
     .
     .	     offset_lo  	( trans_nrst_indx_abv )	/  -1 /,
     .	     offset_hi  	( trans_nrst_indx_abv )	/  +1 /,
     .	     offset_supplied	( trans_nrst_indx_abv )	/ .TRUE. /,
     .
     .	     offset_lo  	( trans_nrst_indx_blw )	/  -1 /,
     .	     offset_hi  	( trans_nrst_indx_blw )	/  +1 /,
     .	     offset_supplied	( trans_nrst_indx_blw )	/ .TRUE. /

* --- end of introductory code ---
* initiailize
	trans = cx_trans    ( idim, cx )
	arg   = cx_trans_arg( idim, cx )
	status = ferr_ok

	IF ( offset_supplied(trans) .AND. arg.NE. bad_val4 ) THEN
* offset given explicitly by user (eg @SHF:-1)
	   ioff = INT( arg )
	   IF ( trans .EQ. trans_smth_box
     .	  . OR. trans .EQ. trans_smth_binml
     .	  . OR. trans .EQ. trans_smth_hanng
     .	  . OR. trans .EQ. trans_smth_parzn
     .	  . OR. trans .EQ. trans_smth_welch
     .	  . OR. trans .EQ. trans_fill_ave ) THEN
	      IF (ioff .LE. 0) GOTO 5100
	      hi_off = ioff / 2
	      lo_off = -hi_off
	   ELSEIF ( trans .EQ. trans_fill_interp
     .         .OR. trans .EQ. trans_fill_near     ) THEN  ! number of points 
	      IF (ioff .LE. 0) GOTO 5100
              hi_off = ioff                                ! past edge
              lo_off = -hi_off
	   ELSEIF ( trans .EQ. trans_shift ) THEN
	      hi_off = ioff
	      lo_off = ioff
	   ELSEIF ( trans .GE. trans_nrst_dist_abv
     .	    .AND.   trans .LE. trans_nrst_indx_blw ) THEN
	      IF (ioff .LE. 0) GOTO 5100
              hi_off = ioff                                ! past edge
              lo_off = -hi_off
	   ELSE
! temp - for debugging
	      WRITE (6,*) 'illegal plane transform', trans
	   ENDIF

	ELSE
* use default or standard offsets
	   lo_off = offset_lo( trans )
	   hi_off = offset_hi( trans )
	ENDIF

	RETURN

* error exits
 5000	RETURN
 5100	buff = VAR_TRANS(idim, cx, lout )
	CALL ERRMSG( ferr_invalid_command, status,
     .               'illegal transform argument: '//buff(:lout),
     .               *5000 )
 
	END
