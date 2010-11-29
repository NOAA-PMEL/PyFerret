
      SUBROUTINE pickout (atim, n, val)


*  ACM 5/2003
*  Called from the ave_scat2grid_t function.
*  Pick out nth item from atim, which is really 4-D 
*  with normal axes on three of the four axes.

      REAL atim(*)
      REAL val
      INTEGER n

      val = atim(n)

      RETURN
      END
