	SUBROUTINE GCF_SET_ALT_FCN_FOR_DIR( gcfcn, idirs, num_dirs)

* Update the function name and function number if the function has an
* argument that specifies its direction. 

*       *acm* 3/12 6D Ferret (common uses nferdims in tmap_dims.parm)

        include 'tmap_dims.parm'
        include 'ferret.parm'
	include 'grid_chg_fcns.parm' 
	include 'xgrid_chg_fcns.cmn'


* calling argument declarations
	INTEGER		gcfcn, idirs(*), num_dirs

* Local declarations
	INTEGER	TM_LENSTR1, GCF_FIND_FCN, nlen, isubst, idim
	CHARACTER name*40, buff*40, axdir(nferdims)*1
        DATA axdir / 'X', 'Y', 'Z', 'T', 'E', 'F' /

	IF (num_dirs .EQ. 0)  GOTO 5100

	name = gfcn_alt_name_base(gcfcn)
	nlen = TM_LENSTR1( name )
	isubst = INDEX(name, '%')
	IF (isubst .EQ. 0) GOTO 5100

	name = gfcn_alt_name_base(gcfcn)(:isubst-1)
	name = gfcn_alt_name_base(gcfcn)(isubst+1:nlen)
	name = gfcn_alt_name_base(gcfcn)(:isubst-1)//axdir(idirs(1))
	name = gfcn_alt_name_base(gcfcn)(:isubst-1)//axdir(idirs(1))//
     .          gfcn_alt_name_base(gcfcn)(isubst+1:nlen)

* This is untested as of 3/29/17
	DO idim = 2, num_dirs
	   buff = name
	   nlen = TM_LENSTR1( buff )
	   isubst = INDEX(buff, '%')
	   IF (isubst .EQ. 0) GOTO 5100
	   buff = name//buff(:isubst-1)//axdir(idirs(idim))//
     .          buff(isubst+1:nlen)
           name = buff
        ENDDO
	   
	nlen = TM_LENSTR1( name )
	gcfcn = GCF_FIND_FCN( name )

	RETURN
5100    gcfcn = unspecified_int4
	RETURN
	END
