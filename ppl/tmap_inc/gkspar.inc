C***********************************************************************
C***********************************************************************
C***********************************************************************
C*		Copyright IBM Corporation 1989
C*
C*                      All Rights Reserved
C*
C* Permission to use, copy, modify, and distribute this software and its
C* documentation for any purpose and without fee is hereby granted,
C* provided that the above copyright notice appear in all copies and that
C* both that copyright notice and this permission notice appear in
C* supporting documentation, and that the name of IBM not be
C* used in advertising or publicity pertaining to distribution of the
C* software without specific, written prior permission.
C*
C* IBM DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
C* ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT SHALL
C* IBM BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
C* ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
C* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
C* ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS
C* SOFTWARE.
C*
C*****                                                             *****
C*****   Product:     graPHIGS GKS-CO Demonstration Program        *****
C*****                                                             *****
C*****   File Name:   gkspar.inc                                   *****
C*****                                                             *****
C*****   Descriptive Name: This file contains mnemonic FORTRAN     *****
C*****                     names and their values for GKS          *****
C*****                     enumeration types. Each enumeration     *****
C*****                     type is a variable initialized          *****
C*****                     through a parameter statement. The      *****
C*****                     last two sections (font types and       *****
C*****                     color table indexes) are not standard   *****
C*****                     GKS FORTRAN enumerations, but were      *****
C*****                     added specifically for use by the       *****
C*****                     GKSDEMO program.                        *****
C*****                                                             *****
C***********************************************************************
C***********************************************************************
C***********************************************************************
 
C***********************************************************************
C*****                                                             *****
C*****   Aspect Source Flag: Bundled, Individual                   *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GBUNDL,GINDIV
      PARAMETER     (GBUNDL = 0, GINDIV = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Attribute Control Flag: Current, Specified                *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GCURNT,GSPEC
      PARAMETER     (GCURNT = 0, GSPEC = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Clear Control Flag: Conditionally, Always                 *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GCONDI,GALWAY
      PARAMETER     (GCONDI = 0, GALWAY = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Clipping Indicator: No Clip, Clip                         *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNCLIP,GCLIP
      PARAMETER     (GNCLIP = 0, GCLIP = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Color Available: Monochrome, Color                        *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GMONOC,GCOLOR
      PARAMETER     (GMONOC = 0, GCOLOR = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Coordinate Switch: WC,NDC                                 *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GWC,GNDC
      PARAMETER     (GWC = 0, GNDC = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Deferral Mode: ASAP, BNIG, BNIL, ASTI                     *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GASAP,GBNIG,GBNIL,GASTI
      PARAMETER     (GASAP = 0, GBNIG = 1, GBNIL = 2, GASTI = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Detectability: Undetectable, Detectable                   *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GUNDET,GDETEC
      PARAMETER     (GUNDET = 0, GDETEC = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Device Coordinate Units: Meters, Other                    *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GMETRE,GOTHU
      PARAMETER     (GMETRE = 0, GOTHU = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Displace Surface Empty: Not Empty, Empty                  *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNEMPT,GEMPTY
      PARAMETER     (GNEMPT = 0, GEMPTY = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Dynamic Modification: Implicit Regeneration, Immediate    *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GIRG,GIMM
      PARAMETER     (GIRG = 0, GIMM = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Echo Switch: No Echo, Echo                                *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNECHO,GECHO
      PARAMETER     (GNECHO = 0, GECHO = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Fill Area Interior Style: Hollow, Solid, Pattern, Hatch   *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GHOLLO,GSOLID,GPATTR,GHATCH
      PARAMETER     (GHOLLO = 0, GSOLID = 1, GPATTR = 2, GHATCH = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   GDP Attributes: Polyline, Polymarker, Text, Fill Area     *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GPLATT, GPMATT, GTXATT, GFAATT
      PARAMETER     (GPLATT = 0, GPMATT = 1, GTXATT = 2, GFAATT = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Highlighting: Normal, Highlighted                         *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNORML,GHILIT
      PARAMETER     (GNORML = 0, GHILIT = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Initial Choice Prompt Flag: Off, On                       *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GPROFF,GPRON
      PARAMETER     (GPROFF = 0, GPRON = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Input Device Status: None, OK, No Pick or No Choice       *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNONE, GOK, GNPICK, GNCHOI
      PARAMETER     (GNONE = 0, GOK = 1, GNPICK = 2, GNCHOI = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Input Class: None, Locator, Stroke, Valuator, Choice,     *****
C*****                Pick, String                                 *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNCLAS,GLOCAT,GSTROK,GVALUA,GCHOIC,GPICK,GSTRIN
      PARAMETER     (GNCLAS = 0, GLOCAT = 1, GSTROK = 2, GVALUA = 3)
      PARAMETER     (GCHOIC = 4, GPICK = 5, GSTRIN = 6)
 
C***********************************************************************
C*****                                                             *****
C*****   Implicit Regeneration Mode: Suppressed, Allowed           *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GSUPPD,GALLOW
      PARAMETER     (GSUPPD = 0, GALLOW = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Level of GKS: 0a, 0b, 0c, 1a, 1b, 1c, 2a, 2b, 2c          *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GL0A,GL0B,GL0C,GL1A,GL1B,GL1C,GL2A,GL2B,GL2C
      PARAMETER     (GL0A = 0, GL0B = 1, GL0C = 2, GL1A = 3, GL1B = 4)
      PARAMETER     (GL1C = 5, GL2A = 6, GL2B = 7, GL2C = 8)
 
C***********************************************************************
C*****                                                             *****
C*****   Line Type: Solid, Dash, Dot, Dash-Dot                     *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GLSOLI,GLDASH,GLDOT,GLDASD
      PARAMETER     (GLSOLI = 1, GLDASH = 2, GLDOT = 3, GLDASD = 4)
 
C***********************************************************************
C*****                                                             *****
C*****   Marker Type: Period, Plus, Asterisk, Circle, Cross        *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GPOINT,GPLUS,GAST,GOMARK,GXMARK
      PARAMETER     (GPOINT = 1, GPLUS = 2, GAST = 3, GOMARK = 4)
      PARAMETER     (GXMARK = 5)
 
C***********************************************************************
C*****                                                             *****
C*****   New Frame Action Necessary: No, Yes                       *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNO,GYES
      PARAMETER     (GNO = 0, GYES = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Operating Mode: Request, Sample, Event                    *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GREQU,GSAMPL,GEVENT
      PARAMETER     (GREQU = 0, GSAMPL = 1, GEVENT = 2)
 
C***********************************************************************
C*****                                                             *****
C*****   Operating State Value: GKCL, GKOP, WSOP, WSAC, SGOP       *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GGKCL,GGKOP,GWSOP,GWSAC,GSGOP
      PARAMETER     (GGKCL = 0, GGKOP = 1, GWSOP = 2, GWSAC = 3)
      PARAMETER     (GSGOP = 4)
 
C***********************************************************************
C*****                                                             *****
C*****   Polyline/Fill Area Control Flag: Polyline, Fill Area      *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GPLINE,GFILLA
      PARAMETER     (GPLINE = 0, GFILLA = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Presence of Invalid Values: Absent, Present               *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GABSNT,GPRSNT
      PARAMETER     (GABSNT = 0, GPRSNT = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Regeneration Flag: Postpone, Perform                      *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GPOSTP,GPERFO
      PARAMETER     (GPOSTP = 0, GPERFO = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Relative Input Priority: Higher, Lower                    *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GHIGHR,GLOWER
      PARAMETER     (GHIGHR = 0, GLOWER = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Simultaneous Events Flag: No More, More                   *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNMORE, GMORE
      PARAMETER     (GNMORE = 0, GMORE = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Text Alignment Horizontal                                 *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GAHNOR,GALEFT,GACENT,GARITE
      PARAMETER     (GAHNOR = 0, GALEFT = 1, GACENT = 2, GARITE = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Text Alignment Vertical: Normal, Top, Cap, Half, Base,    *****
C*****                            Bottom                           *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GAVNOR,GATOP,GACAP,GAHALF,GABASE,GABOTT
      PARAMETER     (GAVNOR = 0, GATOP = 1, GACAP = 2, GAHALF =3)
      PARAMETER     (GABASE = 4, GABOTT = 5)
 
C***********************************************************************
C*****                                                             *****
C*****   Text Path: Right, Left, Up, Down                          *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GRIGHT, GLEFT, GUP, GDOWN
      PARAMETER     (GRIGHT = 0, GLEFT = 1, GUP = 2, GDOWN = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Text Precision: String, Character, Stroke                 *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GSTRP,GCHARP,GSTRKP
      PARAMETER     (GSTRP = 0, GCHARP = 1, GSTRKP = 2)
 
C***********************************************************************
C*****                                                             *****
C*****   Type of Returned Value: Set, Realized                     *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GSET,GREALI
      PARAMETER     (GSET = 0, GREALI = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Update State: Not Pending, Pending                        *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GNPEND,GPEND
      PARAMETER     (GNPEND = 0, GPEND = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Vector / Rastor / Other Type of Display                   *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GVECTR,GRASTR,GOTHWK
      PARAMETER     (GVECTR = 0, GRASTR = 1, GOTHWK = 2)
 
C***********************************************************************
C*****                                                             *****
C*****   Visibility: Visible, Invisible                            *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GINVIS,GVISI
      PARAMETER     (GINVIS = 0, GVISI = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Workstation Category: Output, Input, Input/Output,        *****
C*****                         WISS, Metafile Output, Metafile     *****
C*****                         Input                               *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GOUTPT,GINPUT,GOUTIN,GWISS,GMO,GMI
      PARAMETER     (GOUTPT = 0, GINPUT = 1, GOUTIN = 2, GWISS = 3)
      PARAMETER     (GMO = 4, GMI = 5)
 
C***********************************************************************
C*****                                                             *****
C*****   Workstation State: Inactive, Active                       *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GINACT, GACTIV
      PARAMETER     (GINACT = 0, GACTIV = 1)
 
C***********************************************************************
C*****                                                             *****
C*****   Font Type:  Font 1, Font 2, Font 3                        *****
C*****                                                             *****
C*****     Note: This is not a standard GKS enumeration - it       *****
C*****           was added to the standard list specifically       *****
C*****           for use by the GKSDEMO program example.           *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GFONT1,GFONT2,GFONT3
      PARAMETER     (GFONT1 = 1, GFONT2 = 2, GFONT3 = 3)
 
C***********************************************************************
C*****                                                             *****
C*****   Color Table Indexes as Set by GKSDEMO Program             *****
C*****                                                             *****
C*****     Note: This is not a standard GKS enumeration - it       *****
C*****           was added to the standard list specifically       *****
C*****           for use by the GKSDEMO program example.           *****
C*****                                                             *****
C***********************************************************************
 
      INTEGER*4     GBLACK,GRED,GGREEN,GBLUE,GYELOW,GORNGE,GMAGNT
      INTEGER*4     GLMGNT,GLBLUE,GLTURQ,GGRAYB,GLGREN,GMGREN
      INTEGER*4     GMRED,GGRAY,GWHITE
      PARAMETER     (GBLACK = 0, GRED = 1, GGREEN = 2, GBLUE = 3)
      PARAMETER     (GYELOW = 4, GORNGE = 5, GMAGNT = 6)
      PARAMETER     (GLMGNT = 7, GLBLUE = 8, GLTURQ = 9)
      PARAMETER     (GGRAYB = 10, GLGREN = 11, GMGREN = 12)
      PARAMETER     (GMRED = 13, GGRAY = 14, GWHITE = 15)
 
