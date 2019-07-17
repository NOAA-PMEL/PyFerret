	SUBROUTINE KEY_SET_UP (string_labs, skip_key, max_len, 
     .     lab_digits, check_0, lab_add, num_dec, lev1, levn, incr)


	include 	'parampl5_dat.decl'
	include 	'PARAMPL5.DAT'
	include		'cont_inc.decl'
	include		'CONT.INC'
        include 	'pltcom_dat.decl'
        include 	'PLTCOM.DAT'
        include 	'pltl_inc.decl'
        include 	'PLTL.INC'

        include         'axis_inc.decl'
        include         'AXIS.INC'
        include         'plt_inc.decl'
        include         'PLT.INC'
        include         'pen_inc.decl'
        include         'PEN.INC'

        include         'shade_vars.cmn'
        include         'gkscm1_inc.decl'
        include         'GKSCM1.INC'
        include         'gkscm2.cmn'

	include        'gkspar.inc'

* arguments
	LOGICAL string_labs, skip_key, check_0
	INTEGER max_len, lab_digits, num_dec, lev1, levn, incr

	LOGICAL TM_FPEQ_SNGL, TM_HAS_STRING, 
     .		have_neg, increase_digits
	INTEGER TM_LENSTR1, lev_digits, str_len, have_exp
	REAL	del, val
        REAL*8	val_n, val_last
	CHARACTER*48 TM_FMT_SNGL, buff, text

* Get max length for key labels. 

* initialize
        del = 1
        check_0 = .TRUE.

        lev1 = 1
        levn = nlev2
	incr = 1
        lab_add = 0
        num_dec = 0
        max_len = 0

	skip_key = (levn .EQ. 1 .AND. zlev(1) .GE. 1.e33)
	IF (skip_key) GOTO 5000

* max_len used for horizontal axis computations

	IF (string_labs) THEN
           max_len = 0
	   DO ndx = lev1, shd_levels
	      buff = colorkey_labs(lev1)
	      max_len = MAX (max_len, TM_LENSTR1(buff))   
	   ENDDO
	   
	   GOTO 5000
	ENDIF

* Decide whether to check for small numeric values in the key labels.
* If delta is large, we dont want 0.000005 to be labelled in an exponential
* format (it is zero with a bit of noise). But if all the numbers are 
* small, label them as is. 

	IF (levn.EQ.1) ZLEV(2) = zlev(1)
        IF (neginf) lev1 = 2
        IF (posinf) levn = nlev2 - 1
        IF (neginf.AND.posinf .AND.shd_levels.EQ.1) levn = nlev2

        IF (shd_levels .GT. 1) del = zlev(lev1+1) - zlev(lev1)
        IF (TM_FPEQ_SNGL(del,0.)) check_0 = .FALSE.
        
* See if klab_dig is enough digits to distinguish the levels. 

        lab_digits = klab_dig
        increase_digits = .TRUE.

	if (klab_incr .eq. 0) then
	   incr = (shd_levels/25) + 1
	else
	   incr = klab_incr
	end if

	kount = 0
        DO WHILE (increase_digits .AND. ABS(lab_digits) .LE. klab_len .AND.kount.LT.10)
           buff = ' '
           val = zlev(lev1)
	   IF (center_key_labels) val = zlevsave(lev1)
           IF (TM_FPEQ_SNGL(val,0.) .AND. check_0) val = 0.
 	   text = TM_FMT_SNGL(val,lab_digits,klab_len+5,str_len)

           increase_digits = .FALSE.

 	   have_exp = 0
 	   IF (INDEX(buff, 'E') .GT.0) have_exp = 1
 	   nl = 1

           num_dec = 0
	   have_neg = .FALSE.
           max_len = 0

	   DO 30 ndx = lev1+incr, shd_levels, incr
             val = zlev(ndx)
             IF (center_key_labels) THEN
	        IF (ndx.GE.shd_levels) EXIT ! from loop
		val = zlevsave(ndx)
             ENDIF

	     IF (TM_FPEQ_SNGL(val,0.) .AND. check_0) val = 0.
	     buff = TM_FMT_SNGL (val,lab_digits,klab_len+5,str_len)

	     increase_digits = (buff .EQ. text)   ! Are labels different, one to the next?
             text = buff

             val_last = val_n
             IF (max_len .lt. str_len) max_len = str_len

	     i = INDEX(buff, '.')
             IF (i .GT. 0) num_dec = MAX(num_dec, str_len-i)
	     have_neg = have_neg .OR. val.LT.0 
	     
             nl = nl + 1
             IF (INDEX(buff, 'E') .GT.0) have_exp = have_exp + 1

   30      CONTINUE

           IF (increase_digits) THEN
              IF (lab_digits .GT. 0) lab_digits = lab_digits + 1
              IF (lab_digits .LT. 0) lab_digits = lab_digits - 1
              lab_add = lab_add + 1
           ENDIF

* If some labels are formatted as exponentials and some not, try
* using fewer digits for all exponential.
           IF (have_exp .GT. 0 .AND. .NOT.increase_digits) lab_digits = lab_digits - 1

           kount = kount + 1
        ENDDO

 5000	RETURN
	END
