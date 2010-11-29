/*
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
*/



/* wgif.c - containing 

   void wGIF(FILE *fp, XImage *image,int r[],int g[], int b[]);
and
  GIFEncode( fp, GHeight, GWidth, GInterlace, Background,
             BitsPerPixel, Red, Green, Blue, GetPixel )
   

Note: for **unknown reasons** this routine will not compile using the stock
"cc" command on SunOS.  We use gcc instead.

    gcc -g -c wgif.c

 NOAA/PMEL, Seattle, WA - Tropical Modeling and Analysis Program

 Nov. '94 - Kevin O'Brien based on xpaint

*kob* 5/96 - modified slightly to change declaration of data array from
             character to unsigned character.  This solved a problem that
	     occurred when attempting to save a gif file when hi pixel/color
	     values were required (ie, if using a lot of colors

*kob* 6/12/96 - explicit cast of image->data to eliminate compiler complaints
*acm* 6/18/07 - renamed compress to wcompress because of conflict under x86-64_linux 
                with /usr/lib64/libz.a(compress.o)

 Routine for writing out GIF files, using pd GIFEncode routine */

/* *kob* 10/03 v553 - gcc v3.x needs wchar.h included */
#include <wchar.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <signal.h>
#ifdef MEMDBG
#include <mnemosyne.h>
#endif
#include <X11/Xlib.h>

typedef int code_int;                   /* must be able to hold values -1 to 2**BITS */
 
#ifdef SIGNED_COMPARE_SLOW
typedef unsigned long int count_int;
typedef unsigned short int count_short;
#else
typedef long int          count_int;
#endif
 
#ifdef NO_UCHAR
typedef char   char_type;
#else
typedef        unsigned char   char_type;
#endif /* UCHAR */
 
#define TRUE 1
#define FALSE 0
 
#define BITS    12
#define HSIZE  5003            /* 80% occupancy */
/**************************************************************************/
/* static char_type			*data; */
static char_type			*data;
static int				iwidth, iheight, image_offset;
/**************************************************************************/
typedef int (* ifunptr)();                              /* Pointer to function returning an int */
 
int GetPixel( int, int );

static Putword( int, FILE *);
static cl_block ();   
static cl_hash(register count_int);
static writeerr();
static char_init();
static char_out( int );
static flush_char();
static output( code_int );



void wGIF(fp, image,r,g,b)
     FILE *fp;
     XImage *image;
     int r[],g[],b[];
{

     int x,y;

/* Set global variables needed for GetPixel routine */
/* cast image-> data to unsigned character ptr. 6/12/96 *kob* */
        data = ( char_type *) image->data;
	iwidth = image->bytes_per_line;
	iheight = image->height;

        for (x=0; x < (1 << image->bits_per_pixel) ; x++)
	  {
	    r[x] =r[x]   >> 8;
	    g[x]=g[x] >> 8;
	    b[x]=b[x]   >> 8;
	  }


/* write out GIF file */
	GIFEncode(fp, image->width, image->height, 0, image->depth, image->bits_per_pixel, r, g, b, GetPixel);

	fclose(fp);

}
 
 
 
/*****************************************************************************
 *
 * GIFENCODE.C    - GIF Image compression interface
 *
 * GIFEncode( fp, GHeight, GWidth, GInterlace, Background,
 *            BitsPerPixel, Red, Green, Blue, GetPixel )
 *
 *****************************************************************************/
 
static int Width, Height;
static int curx, cury;
static long CountDown;
static int Pass = 0;
static int Interlace;
 
static BumpPixel() /* Bump the 'curx' and 'cury' to point to the next pixel */
{
        curx++;                 /* Bump the current X position */
 
/* If we are at the end of a scan line, set curx back to the beginning.
        If we are interlaced, bump the cury to the appropriate spot,
        otherwise, just increment it.
*/
 
        if( curx == Width )
                {
                        curx = 0;
 
                        if( !Interlace )
                                cury++;
                        else
                                {
                                        switch( Pass )
                                                {
                                                        case 0:
                                                                cury += 8;
                                                                if( cury >= Height )
                                                                        {
                                                                                Pass++;
                                                                                cury = 4;
                                                                        }
                                                                break;
 
                                                        case 1:
                                                                cury += 8;
                                                                if( cury >= Height )
                                                                        {
                                                                                Pass++;
                                                                                cury = 2;
                                                                        }
                                                                break;
 
                                                        case 2:
                                                                cury += 4;
                                                                if( cury >= Height )
                                                                        {
                                                                                Pass++;
                                                                                cury = 1;
                                                                        }
                                                                break;
 
                                                        case 3:
                                                                cury += 2;
                                                                break;
                                                }
                                }
                }
}
 
 
 
 
 
GIFGetPixel( getpixel )                        /* Return the next pixel from the image */
ifunptr getpixel;
{
        int r;
 
        if( CountDown == 0 )
                return EOF;
 
        CountDown--;
 
        r = ( * getpixel )( curx, cury );
 
        BumpPixel();
 
        return r;
}
 
 
 
 
GIFEncode( fp, GWidth, GHeight, GInterlace, Background,
           BitsPerPixel, Red, Green, Blue, GetPixel )
 
FILE *fp;
int GWidth;
int GHeight;
int GInterlace;
int Background;
int BitsPerPixel;
int Red[], Green[], Blue[];
ifunptr GetPixel;
 
{
        int B;
        int RWidth, RHeight;
        int LeftOfs, TopOfs;
        int Resolution;
        int ColorMapSize;
        int InitCodeSize;
        int i;
 
        Interlace = GInterlace;
 
        ColorMapSize = 1 << BitsPerPixel;
 
        RWidth = Width = GWidth;
        RHeight = Height = GHeight;
        LeftOfs = TopOfs = 0;
 
        Resolution = BitsPerPixel;
 
        CountDown = (long)Width * (long)Height; /* Calculate number of bits */
 
        Pass = 0;                                       /* Indicate which pass we are on (if interlace) */
 
        if( BitsPerPixel <= 1 )         /* The initial code size */
                InitCodeSize = 2;
        else
                InitCodeSize = BitsPerPixel;
 
        curx = cury = 0;                                /* Set up the current x and y position */
 
        fwrite( "GIF87a", 1, 6, fp );           /* Write the Magic header */
 
        Putword( RWidth, fp );                  /* Write out the screen width and height */
        Putword( RHeight, fp );
 
        B = 0x80;       /* Yes, there is a color map */
        B |= (Resolution - 1) << 5;      /* OR in the Resolution                */
        B |= (BitsPerPixel - 1);                        /* OR in the Bits per Pixel */
 
        fputc( B, fp );                                         /* Write it out */
 
        fputc( Background, fp );                        /* Write out the Background colour */
 
        fputc( 0, fp );                                         /* Byte of 0's (future expansion) */
 
 
        for( i=0; i<ColorMapSize; i++ )         /* Write out the Global Colour Map */
                {
                        fputc( Red[i], fp );
                        fputc( Green[i], fp );
                        fputc( Blue[i], fp );
                }
 
        fputc( ',', fp );                                               /* Write an Image separator */
 
        Putword( LeftOfs, fp );                         /* Write the Image header */
        Putword( TopOfs, fp );
        Putword( Width, fp );
        Putword( Height, fp );
 
        if( Interlace )         /* Write out whether or not the image is interlaced */
                fputc( 0x40, fp );
        else
                fputc( 0x00, fp );
 
        fputc( InitCodeSize, fp );                      /* Write out the initial code size */
 
 /* renamed from compress because of conflict under x86-64_linux with  /usr/lib64/libz.a(compress.o) */
        wcompress( InitCodeSize+1, fp, GetPixel );               /* Actually compress data */
 
 
 
        fputc( 0, fp );                                         /* Write out Zero-length(to end series) */
        fputc( ';', fp );                                       /* Write the GIF file terminator */

}
 
 
 
 
static Putword( w, fp )                                 /* Write out a word to the GIF file */
int w;
FILE *fp;
{
        fputc( w & 0xff, fp );
        fputc( (w / 256) & 0xff, fp );
}
 
 
 
 
/***************************************************************************
 *
 *  GIFENCOD.C       - GIF Image compression routines
 *
 *  Lempel-Ziv compression based on 'compress'.  GIF modifications by
 *  David Rowley (mgardi@watdcsu.waterloo.edu)
 *
 ***************************************************************************/
 
/*
 *
 * GIF Image compression - modified 'compress'
 *
 * Based on: compress.c - File compression ala IEEE Computer, June 1984.
 *
 * By Authors:  Spencer W. Thomas       (decvax!harpo!utah-cs!utah-gr!thomas)
 *              Jim McKie               (decvax!mcvax!jim)
 *              Steve Davies            (decvax!vax135!petsd!peora!srd)
 *              Ken Turkowski           (decvax!decwrl!turtlevax!ken)
 *              James A. Woods          (decvax!ihnp4!ames!jaw)
 *              Joe Orost               (decvax!vax135!petsd!joe)
 *
 */
 
#define ARGVAL() (*++(*argv) || (--argc && *++argv))
 
static int n_bits;                        /* number of bits/code */
static int maxbits = BITS;                /* user settable max # bits/code */
static code_int maxcode;                  /* maximum code, given n_bits */
static code_int maxmaxcode = (code_int)1 << BITS; /* should NEVER generate this code */
#ifdef COMPATIBLE               /* But wrong! */
# define MAXCODE(n_bits)        ((code_int) 1 << (n_bits) - 1)
#else
# define MAXCODE(n_bits)        (((code_int) 1 << (n_bits)) - 1)
#endif /* COMPATIBLE */
 
static count_int htab [HSIZE];
static unsigned short codetab [HSIZE];
#define HashTabOf(i)       htab[i]
#define CodeTabOf(i)    codetab[i]
 
static code_int hsize = HSIZE;                 /* for dynamic table sizing */
static count_int fsize;
 
/*
 * To save much memory, we overlay the table used by compress() with those
 * used by decompress().  The tab_prefix table is the same size and type
 * as the codetab.  The tab_suffix table needs 2**BITS characters.  We
 * get this from the beginning of htab.  The output stack uses the rest
 * of htab, and contains characters.  There is plenty of room for any
 * possible stack (stack used to be 8000 characters).
 */
 
#define tab_prefixof(i) CodeTabOf(i)
#define tab_suffixof(i)        ((char_type *)(htab))[i]
#define de_stack               ((char_type *)&tab_suffixof((code_int)1<<BITS))
 
static code_int free_ent = 0;                  /* first unused entry */
static int exit_stat = 0;
 
/*
 * block compression parameters -- after all codes are used up,
 * and compression rate changes, start over.
 */
static int clear_flg = 0;
 
static int offset;
static long int in_count = 1;            /* length of input */
static long int out_count = 0;           /* # of codes output (for debugging) */
 
/*
 * compress stdin to stdout
 *
 * Algorithm:  use open addressing double hashing (no chaining) on the
 * prefix code / next character combination.  We do a variant of Knuth's
 * algorithm D (vol. 3, sec. 6.4) along with G. Knott's relatively-prime
 * secondary probe.  Here, the modular division first probe is gives way
 * to a faster exclusive-or manipulation.  Also do block compression with
 * an adaptive reset, whereby the code table is cleared when the compression
 * ratio decreases, but after the table fills.  The variable-length output
 * codes are re-sized at this point, and a special CLEAR code is generated
 * for the decompressor.  Late addition:  construct the table according to
 * file size for noticeable speed improvement on small files.  Please direct
 * questions about this implementation to ames!jaw.
 */
 
static int g_init_bits;
static FILE *g_outfile;
 
static int ClearCode;
static int EOFCode;
 
wcompress( init_bits, outfile, ReadValue )
int init_bits;
FILE *outfile;
ifunptr ReadValue;
{
    register long fcode;
    register code_int i = 0;
    register int c;
    register code_int ent;
    register code_int disp;
    register code_int hsize_reg;
    register int hshift;
 
         g_init_bits = init_bits;               /* g_init_bits - initial number of bits */
         g_outfile = outfile;                   /* g_outfile   - pointer to output file */
 
         offset = 0;                                            /* Set up the necessary values */
         out_count = 0;
         clear_flg = 0;
         in_count = 1;
         maxcode = MAXCODE(n_bits = g_init_bits);
 
    ClearCode = (1 << (init_bits - 1));
    EOFCode = ClearCode + 1;
    free_ent = ClearCode + 2;
 
    char_init();
 
    ent = GIFGetPixel( ReadValue );
 
    hshift = 0;
    for ( fcode = (long) hsize;  fcode < 65536L; fcode *= 2L )
        hshift++;
    hshift = 8 - hshift;                /* set hash code range bound */
 
    hsize_reg = hsize;
    cl_hash( (count_int) hsize_reg);            /* clear hash table */
 
         output( (code_int)ClearCode );
/***************************************************************************/
#ifdef SIGNED_COMPARE_SLOW
         while( (c = GIFGetPixel( ReadValue )) != (unsigned) EOF )
                {
#else
         while( (c = GIFGetPixel( ReadValue )) != EOF )
                {
#endif
 
                  in_count++;
 
        fcode = (long) (((long) c << maxbits) + ent);
        i = (((code_int)c << hshift) ^ ent);    /* xor hashing */
 
        if ( HashTabOf (i) == fcode ) {
            ent = CodeTabOf (i);
            continue;
        } else if ( (long)HashTabOf (i) < 0 )      /* empty slot */
            goto nomatch;
        disp = hsize_reg - i;           /* secondary hash (after G. Knott) */
        if ( i == 0 )
            disp = 1;
probe:
        if ( (i -= disp) < 0 )
            i += hsize_reg;
 
        if ( HashTabOf (i) == fcode ) {
            ent = CodeTabOf (i);
            continue;
        }
        if ( (long)HashTabOf (i) > 0 )
            goto probe;
nomatch:
        output ( (code_int) ent );
        out_count++;
        ent = c;
#ifdef SIGNED_COMPARE_SLOW
        if ( (unsigned) free_ent < (unsigned) maxmaxcode) {
#else
        if ( free_ent < maxmaxcode ) {
#endif
            CodeTabOf (i) = free_ent++; /* code -> hashtable */
            HashTabOf (i) = fcode;
        } else
                cl_block();
    }
    /*
     * Put out the final code.
     */
    output( (code_int)ent );
    out_count++;
    output( (code_int) EOFCode );
 
    return;
}
 
 
 
 
 
 
/*****************************************************************
 * TAG( output )
 *
 * Output the given code.
 * Inputs:
 *      code:   A n_bits-bit integer.  If == -1, then EOF.  This assumes
 *              that n_bits =< (long)wordsize - 1.
 * Outputs:
 *      Outputs code to the file.
 * Assumptions:
 *      Chars are 8 bits long.
 * Algorithm:
 *      Maintain a BITS character long buffer (so that 8 codes will
 * fit in it exactly).  Use the VAX insv instruction to insert each
 * code in turn.  When the buffer fills up empty it and start over.
 */
 
static unsigned long cur_accum = 0;
static int  cur_bits = 0;
 
static
unsigned long masks[] = { 0x0000, 0x0001, 0x0003, 0x0007, 0x000F,
                                  0x001F, 0x003F, 0x007F, 0x00FF,
                                  0x01FF, 0x03FF, 0x07FF, 0x0FFF,
                                  0x1FFF, 0x3FFF, 0x7FFF, 0xFFFF };
 
 
 
 
static output( code )
code_int  code;
{
    cur_accum &= masks[ cur_bits ];
 
    if( cur_bits > 0 )
        cur_accum |= ((long)code << cur_bits);
    else
        cur_accum = code;
 
    cur_bits += n_bits;
 
    while( cur_bits >= 8 ) {
        char_out( (unsigned int)(cur_accum & 0xff) );
        cur_accum >>= 8;
        cur_bits -= 8;
    }
 
    /*
     * If the next entry is going to be too big for the code size,
     * then increase it, if possible.
     */
   if ( free_ent > maxcode || clear_flg ) {
 
            if( clear_flg ) {
 
                maxcode = MAXCODE (n_bits = g_init_bits);
                clear_flg = 0;
 
            } else {
 
                n_bits++;
                if ( n_bits == maxbits )
                    maxcode = maxmaxcode;
                else
                    maxcode = MAXCODE(n_bits);
            }
        }
 
    if( code == EOFCode ) {
        /*
         * At EOF, write the rest of the buffer.
         */
        while( cur_bits > 0 ) {
                char_out( (unsigned int)(cur_accum & 0xff) );
                cur_accum >>= 8;
                cur_bits -= 8;
        }
 
        flush_char();
 
        fflush( g_outfile );
 
        if( ferror( g_outfile ) )
                writeerr();
	cur_accum = 0;
	cur_bits = 0;
    }
}
 
 
 
static cl_block ()             /* table clear for block compress */
{
 
        cl_hash ( (count_int) hsize );
        free_ent = ClearCode + 2;
        clear_flg = 1;
 
        output( (code_int)ClearCode );
}
 
 
 
static cl_hash(hsize)                                                                   /* reset code table */
register count_int hsize;
{
 
        register count_int *htab_p = htab+hsize;
 
        register long i;
        register long m1 = -1;
 
        i = hsize - 16;
        do {                            /* might use Sys V memset(3) here */
                *(htab_p-16) = m1;
                *(htab_p-15) = m1;
                *(htab_p-14) = m1;
                *(htab_p-13) = m1;
                *(htab_p-12) = m1;
                *(htab_p-11) = m1;
                *(htab_p-10) = m1;
                *(htab_p-9) = m1;
                *(htab_p-8) = m1;
                *(htab_p-7) = m1;
                *(htab_p-6) = m1;
                *(htab_p-5) = m1;
                *(htab_p-4) = m1;
                *(htab_p-3) = m1;
                *(htab_p-2) = m1;
                *(htab_p-1) = m1;
                htab_p -= 16;
        } while ((i -= 16) >= 0);
 
        for ( i += 16; i > 0; i-- )
                *--htab_p = m1;
}
 
 
 
 
static writeerr()
{
        printf( "error writing output file\n" );
        exit(1);
}
 
 
/******************************************************************************
 *
 * GIF Specific routines
 *
 ******************************************************************************/
static int a_count;                     /* Number of characters so far in this 'packet' */
 
 
 
static char_init()                      /* Set up the 'byte output' routine */
{
        a_count = 0;
}
 
 
 
static char accum[ 256 ];       /* Define the storage for the packet accumulator */
 
 
 
static char_out( c )                    /* Add character to end, if 254 characters, flush */
int c;
{
        accum[ a_count++ ] = c;
        if( a_count >= 254 )
                flush_char();
}
 
 
 
static flush_char() /* Flush the packet to disk, and reset the accumulator */
{
        if( a_count > 0 )
                {
                        fputc( a_count, g_outfile );
                        fwrite( accum, 1, a_count, g_outfile );
                        a_count = 0;
                }
}
 
/*************************************************************************/
int GetPixel( x, y )
int x, y;
{
      return data[ y*iwidth + x];
}
/************************/

