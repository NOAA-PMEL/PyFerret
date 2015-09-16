
c***********************************************************************
c...NOBS.....Based on gausswt, return only the # of obs in (x,y) gridbox
c............method is to call this sub for each data value
c............all calcs done in gridbox units
c............xx/x1/xf all in same units
c............yy/y1/yf all in same units

c  i	xx,yy,tt=x/y/t location of data pt (data units)
c  i	nx,ny,nt=size of grids
c  i	xaxlo,xaxyi, etc bounds of destination grid cells
c  o 	obs(nx,ny,nt) number of obs in each (x,y,t) gridbox
c--------------------------------------------------------------------------

	SUBROUTINE nobsxytv (obs_only, xx, yy, tt, vv, nx, ny, nt, 
     .             xaxlo,xaxhi, yaxlo,yaxhi, taxlo,taxhi, indx, obs)


* Argument declarations

        INTEGER nx, ny, nt, obs_only, indx(4,2)
	REAL xx, yy, tt, vv, xaxlo(*), xaxhi(*), 
     .       yaxlo(*), yaxhi(*), taxlo(*), taxhi(*), obs(nx,ny,2,nt)

* Local variable declarations

        INTEGER ig, jg, Lg

	lg = 1
	DO l = 1, nt
	   IF (tt.GE.taxlo(lg) .AND.
     .         tt.LT.taxhi(lg)) THEN
              jg = 1
	      DO j = 1, ny
		 IF (yy.GE.yaxlo(jg) .AND.
     .	             yy.LT.yaxhi(jg)) THEN
                    ig = 1
		    DO i = 1, nx
		       IF ( xx.GE.xaxlo(ig) .AND.
     .                      xx.LT.xaxhi(ig)) THEN

		          IF (obs_only .EQ. 0) THEN
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + vv
		             obs(ig,jg,2,Lg) = obs(ig,jg,2,Lg) + 1.
		          ELSE
		             obs(ig,jg,1,Lg) = obs(ig,jg,1,Lg) + 1.
		             ENDIF

* Store the min and max index used in this cruise
		             indx(1,1) = MIN(indx(1,1), ig)
		             indx(1,2) = MAX(indx(1,2), ig)
		             indx(2,1) = MIN(indx(2,1), jg)
		             indx(2,2) = MAX(indx(2,2), jg)
		             indx(4,1) = MIN(indx(4,1), Lg)
		             indx(4,2) = MAX(indx(4,2), Lg)
		       ENDIF !xx in grid box
		       ig = ig + 1
		    ENDDO  ! nx

		 ENDIF !yy in grid box
		 jg = jg + 1
	      ENDDO  ! ny

	   ENDIF !tt in grid box
	   lg = lg + 1
	ENDDO  ! nt

	RETURN
	END



