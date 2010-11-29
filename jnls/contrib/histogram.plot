CANCEL MODE VERIFY ! histogram.plot - plot the histogram as overlaid viewports
cancel region
! minor change (uses $1 now) 4/94

! produce a fully documented histogram using overlaid viewports to obtain
! labels and titles
cancel viewports                          ! erase previous drawings
define viewport hvp1    ! just like "full"
define viewport hvp2    ! just like "full"
set view hvp1

set data/restore   ! the users original data set
plot/vs/set/@husr $1,$1

! plot only the desired labels - use x axis label as plot title
ppl yaxis -10100,-10000,100
ppl title .15,@AC'LABX'
ppl xlab
! ylab located wrong on the first plot, ok on second try - don't know why
ppl ylab Probability Density Function
ppl axset 0,0,0,0
ppl plot

! restore normal plot style
ppl pen 1,1
ppl axset 1,1,1,1

! now overlay the histogram on these labels
set viewport hvp2
set data ferret_histo.sort
histplot    ! normally "plot/vs/nolab hval,histo[i=@sbx:11]"

! restore user defaults
set region husr
set data/restore

say Use CANCEL VIEWPORTS to erase the histogram plot
SET MODE/LAST VERIFY