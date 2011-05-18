
c***********************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	x1,y1,L1=west/south/earliest edge of grid (center of 1st box in data units)
c  o 	obs(nx,ny,nt) number of obs in each (x,y,t) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobsxytv (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .             x1, y1, t1, dx, dy, dt, xf, yf, tf, indx, obs)


* Argument declarations

        INTEGER nx, ny, nt, obs_only, indx(4,2)
	REAL xx, yy, tt, vv, x1, y1, t1, dx, dy, dt, xf, 
     .       yf, tf, obs(nx,ny,2,nt)

* Local variable declarations

        REAL xxg, yyg, ttg
        INTEGER ig, jg, Lg ! !!!!!!!!!!!!!!!!!!!!!!!! adding t dimension throughtout

	xxg = (xx-x1)/ dx + 1.		  ! grid values of data location
	yyg = (yy-y1)/ dy + 1.
	ttg = (tt-t1)/ dt + 1.

c.................sum the value and count in each (x,y,t) gridbox
	ig = NINT(xxg)
	jg = NINT(yyg)
	Lg = NINT(ttg)
	IF (ig.LT.1 .OR. ig.GT.nx) RETURN
	IF (ig.LT.1 .OR. ig.GT.nx) RETURN
	IF (lg.LT.1 .OR. lg.GT.nt) RETURN

	IF (obs_only .EQ. 0) THEN
           IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .       Lg.GE.1 .AND. Lg.LE.nt   ) THEN
     		obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + vv
     		obs(ig,jg,2,Lg) = obs(ig,jg,2,Lg) + 1.
	   ENDIF
	ELSE
	   IF (ig.GE.1 .AND. ig.LE.nx .AND. jg.GE.1 .AND. jg.LE.ny .AND.
     .      Lg.GE.1 .AND. Lg.LE.nt   ) THEN
     		obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + 1.
	   ENDIF
	ENDIF

* Store the min and max index used in this cruise
	indx(1,1) = MIN(indx(1,1), ig)
	indx(1,2) = MAX(indx(1,2), ig)
	indx(2,1) = MIN(indx(2,1), jg)
	indx(2,2) = MAX(indx(2,2), jg)
	indx(4,1) = MIN(indx(4,1), Lg)
	indx(4,2) = MAX(indx(4,2), Lg)

	RETURN
	END



