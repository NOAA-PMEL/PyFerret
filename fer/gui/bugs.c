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



You can eliminate
>         $(OZDIR)/xgksmod_xopws.c \
> from your Makefile and get rid of the
>       ld:
> /home/e1/tmap/ferret_link/sun/ppl/modsxgks/xgksmod_xopws.o:xxx-multiply defin$
> 
> messages.

Fixed.

        1) I was still unable to change the T index of 12 to 48 in the Coads
climatology.  Upon hitting PLOT the limit returned to 12.
 
        2) The window system lost track of the need to perform updates along
the way.  Buttons, labels, etc. disappeared until obscurred by a window and
redrawn.  I **think** this may have coincided with manually resizing the output
window ... but I can't really be sure what triggered it.

 The full map appears as a momentary flicker before dissappearing as the main
> interface comes up -- that wasn't happening before.

Not sure why this is happening to you and not to me!

> I accidentally opened a bogus data set as my first action (I was in my /tmp  
> directory).  The error dialogs were imperfect because of the stuff I need to 
> fix in the TMAP library.  But then I tried to open a valid data set and got a
> core dump

> Restarted.  In selecting an untested data set (clim_airt_lev) I realized that
> one of the basic bits of missing information is the title and units of the
> variables in a data set.  Can you think of a place to insert an "Info ..."
> button that could perhaps simply execute a "SHOW DATA/VARIABLES" command?  Or
> would that output be too funky?

 Then I corrected this problem by selecting "T Index" and a range of L=1:30 and
> my plot came back as "All Data Are missing" (the error dialog worked great!!!$
> The reason they were all missing was because the point value for Y defaulted $
> the north extreme.  (Even though the scroll bar says it is at the middle.) 
>  There is also some inconsistency between scroll bar and text on the X axis
> when a YT view is selected. n$

Fixed.

> submit the request.  For that matter the "Open Dataset" button remains
> desensitized.


> I tried the custom levels in the "Plot Options ..." menu and the custom levels
> seem to be ignored (actually I see in the MACRO Manager that "/LEVELS" was
> generated but not with no levels specified.)

I wasn't able to duplicate this exactly, but I think the problem came from not having
the update of the 2d plot options tied to a losing focus message.  I automatically hit
return after entering a number which does update the options.  This will probably fix
the problem.

> Finally, I hit "Cancel" in the plot options interface and then requested the
> interface back again from the Main Interface.  I wanted to see if it was fast$
> on the subsequent request. Unfortunately that got me a core dump.

Fixed.

>

> We have 5, 20 and 60 minute resolution ETOPO sets (could make others).  To ma$
> the solid land look good there should be, say, on the order of 10,000 grid
> cells on the screen.  Perhaps a minimum of 7500 cells.  So you look at the X
> and Y limits in degrees on the current plot.
>    
>       square_degrees = xrange * yrange
> 
>       if (square_degrees >  7500 ) then
>               GO fland 60
>       elseif ( square_degrees > 7500/9 ) then     ! "9":  20 min = 1/3 deg
>               GO fland 20
>       elseif ( square_degrees > 7500/36 ) then
>               GO fland 10
>       else
>               GO fland 5

Added.

P.S. I used the data set selector and scrolled until I found "etopo"  (beats me
what directory the file is found in).  So I manually appended "60" to the name
and hit the "Open Dataset" button.  Well the GUI actually sent the command "SET
DATA etopo" instead of "SET DATA etopo60".  BTW the main menu went haywire at
the error -- all the region selectors contain nonsense.

Possibly the GUI should send
         SHOW DATA/VARIABLES selected_dset
where selected_dset is from the current variable selected in the upper left.
 That way info about only the selected data set will be displayed.
 
Fixed.



