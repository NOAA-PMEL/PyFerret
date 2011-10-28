*
* Inquire current (normalization) transformation number
*

        SUBROUTINE FGD_GQCNTN(istat, transnum)

        IMPLICIT NONE
        INTEGER istat, transnum

        INCLUDE 'fgrdel.cmn'

        transnum = activetransform
        istat = 0

        RETURN
        END

