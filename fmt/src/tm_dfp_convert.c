/* tm_dfp_convert_:
/* convert VAX D or IEEE big/little endian double precision floating point */
/* into the currently active CPU representation */
/*** use the pre-processor to select the target word type ***   */

/* *sh* - home brewed */
/* rev. 0.0 2/14/92
/* note: "_" is appended to TM_DFP1_CNVRT by f77 when calling this */
/* replaced "elif" syntax with
	else
	   if
  for SGI port	 - kob 4/8/92 */

/* added ifdef check for underscore in routine name for aix *kob* 10/94 */
#define cptype_vax 0
#define cptype_dec 1
#define cptype_sun 2

#ifdef NO_ENTRY_NAME_UNDERSCORES
void tm_dfp_convert( dval, author_cpu )
#else
void tm_dfp_convert_( dval, author_cpu )
#endif
  double *dval;
  int *author_cpu;
  {

/* internal variable declarations:   */
/* e - exponent                      */
/* f - fraction                      */
/* f_shf - bits from f to be shifted */
  char i1_tmp;
  unsigned short int i2_tmp, e, f_shf, ieee_e;

/* masks                                             */
/* smsk        - sign bit mask                       */
/* nsmsk       - sign bit removal mask               */
/* vax_f1_msk  - f bits from INT*2 word 1 of VAX D   */
/* vax_shf_msk - f bits to shift off right hand end  */
  short int smsk=0100000, nsmsk=077777;
  short int vax_f1_msk=0177, vax_shf_msk=07;
        
  union { double dum;
	  unsigned short int i2[4];
	  char i1[8];
	} u;

/* move the double precision word into the union */
  u.dum = *dval;

#ifdef sun
/* SUN is the platform in use   */
  if    ( *author_cpu == cptype_vax )           /* VAX -> SUN */
/* pre-swap the bytes within each 16 bit word */   
    {
    i1_tmp   = u.i1[0];
    u.i1[0]  = u.i1[1];
    u.i1[1]  = i1_tmp;
    i1_tmp   = u.i1[2];
    u.i1[2]  = u.i1[3];
    u.i1[3]  = i1_tmp;
    i1_tmp   = u.i1[4];
    u.i1[4]  = u.i1[5];
    u.i1[5]  = i1_tmp;
    i1_tmp   = u.i1[6];
    u.i1[6]  = u.i1[7];
    u.i1[7]  = i1_tmp;

/* VAX 16-bit word 1 (with sign bit and exponent)   */
    e = ((u.i2[0] & nsmsk)>>7) - 128;
    ieee_e = (e + 1022)<<4;
    i2_tmp = (u.i2[0] & vax_f1_msk)>>3;  /* bits from f that stay in word 1 */
    f_shf  =  u.i2[0] & vax_shf_msk;     /* bits from f that shift to i2[2] */
    u.i2[0]  = (u.i2[0] & smsk) | ieee_e | i2_tmp;

/* 2nd VAX 16-bit word (all f bits - shift right by 3)   */
    i2_tmp = ((u.i2[1])>>3) | f_shf<<13;
    f_shf  = u.i2[1] & vax_shf_msk;
    u.i2[1]  = i2_tmp;

/* 3rd VAX 16-bit word (all f bits - shift right by 3)   */
    i2_tmp = ((u.i2[2])>>3) | f_shf<<13;
    f_shf  = u.i2[2] & vax_shf_msk;
    u.i2[2]  = i2_tmp;

/* 4th VAX 16-bit word 4 (all f bits - right bits drop off end)   */
    u.i2[3]  = ((u.i2[3])>>3) | f_shf<<13;
    }
  else if ( *author_cpu == cptype_dec )         /* DECstation -> SUN */
    {
    i1_tmp   = u.i1[0];
    u.i1[0]  = u.i1[7];
    u.i1[7]  = i1_tmp;
    i1_tmp   = u.i1[1];
    u.i1[1]  = u.i1[6];
    u.i1[6]  = i1_tmp;
    i1_tmp   = u.i1[2];
    u.i1[2]  = u.i1[5];
    u.i1[5]  = i1_tmp;
    i1_tmp   = u.i1[3];
    u.i1[3]  = u.i1[4];
    u.i1[4]  = i1_tmp;
    }
 
#else
#if unix
/* DECstation is the platform in use   */

  if ( *author_cpu == cptype_vax )    /* VAX --> DECstation */
    {
/* VAX 16-bit word 1 (with sign bit and exponent)   */
    e = ((u.i2[0] & nsmsk)>>7) - 128;
    ieee_e = (e + 1022)<<4;
    i2_tmp = (u.i2[0] & vax_f1_msk)>>3;  /* bits from f that stay in word 1 */
    f_shf  =  u.i2[0] & vax_shf_msk;     /* bits from f that shift to i2[2] */
    u.i2[0]  = (u.i2[0] & smsk) | ieee_e | i2_tmp;

/* 2nd VAX 16-bit word (all f bits - shift right by 3)   */
    i2_tmp = ((u.i2[1])>>3) | f_shf<<13;
    f_shf  = u.i2[1] & vax_shf_msk;
    u.i2[1]  = i2_tmp;

/* 3rd VAX 16-bit word (all f bits - shift right by 3)   */
    i2_tmp = ((u.i2[2])>>3) | f_shf<<13;
    f_shf  = u.i2[2] & vax_shf_msk;
    u.i2[2]  = i2_tmp;

/* 4th VAX 16-bit word 4 (all f bits - right bits drop off end)   */
    u.i2[3]  = ((u.i2[3])>>3) | f_shf<<13;

/* post-swap the 16-bit word order   */
    i2_tmp   = u.i2[0];
    u.i2[0]  = u.i2[3];
    u.i2[3]  = i2_tmp;
    i2_tmp   = u.i2[1];
    u.i2[1]  = u.i2[2];
    u.i2[2]  = i2_tmp;
    }
  else if ( *author_cpu == cptype_sun )     /* SUN -> DECstation */
    {
    i1_tmp   = u.i1[0];
    u.i1[0]  = u.i1[7];
    u.i1[7]  = i1_tmp;
    i1_tmp   = u.i1[1];
    u.i1[1]  = u.i1[6];
    u.i1[6]  = i1_tmp;
    i1_tmp   = u.i1[2];
    u.i1[2]  = u.i1[5];
    u.i1[5]  = i1_tmp;
    i1_tmp   = u.i1[3];
    u.i1[3]  = u.i1[4];
    u.i1[4]  = i1_tmp;
    }

#else
/* VAX is the platform in use   */
     /* not yet figgered out */
#endif   /* decstation */
#endif   /* sun */

/* return the value   */
  *dval = u.dum;

  return ;
  }
