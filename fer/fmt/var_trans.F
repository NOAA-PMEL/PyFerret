	CHARACTER*(*) FUNCTION VAR_TRANS( idim, cx, out_len )

* create a string of the form NAME[Z=lo:hi@trns]

* programmer - steve hankin
* NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program
* written for VAX computer under VMS operating system
*
* V200:  7/21/89 - extracted from DIAG_OP
* V230:  11/4/92 - expanded user variable name length

* calling argument declarations:
	INTEGER	idim, cx, out_len

* internal variable declarations:
	INTEGER	  TM_LENSTR1, slen, vlen
	CHARACTER VAR_CODE*24, CX_DIM_STR*48,
     .            var*24, out_buff*80, dim_str*48

#ifdef unix
	include	'ferret_cmn/ferret.parm'
	include	'ferret_cmn/xcontext.cmn'
	include	'ferret_cmn/xtext_info.cmn'
#else
	INCLUDE	'FERRET_CMN:FERRET.PARM'
	INCLUDE	'FERRET_CMN:XCONTEXT.CMN'
	INCLUDE	'FERRET_CMN:XTEXT_INFO.CMN'
#endif

* internal parameter declarations:
	LOGICAL     full_precision
	PARAMETER ( full_precision = .TRUE. )

* initialize
	var = VAR_CODE( cx_category(cx), cx_variable(cx) )
        vlen = TM_LENSTR1( var )

* TRANSFORMATION
* "VAR_NAME[x=lo:hi@TRN:n,D=#]"
	dim_str = CX_DIM_STR( idim, cx, ':',  full_precision, slen )
	WRITE (out_buff,3000) var(:vlen), dim_str(:slen), cx_data_set(cx)
	CALL TM_PACK_STRING( out_buff, out_buff, 1, 80, out_len )

 3000	FORMAT (A,'[',A,',D=',I3,']')

	VAR_TRANS = out_buff

	RETURN
	END
