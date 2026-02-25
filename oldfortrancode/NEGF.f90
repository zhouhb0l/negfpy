!A NEGF framework developed 11 Feb, 2016, by Zhou hangbo

program NEGF
use sizes
use material_para
use force_constants
use running_para
use interfaces
implicit none

double precision :: kx,ky,kk(2)
double precision :: ww

double complex,allocatable :: sgL(:,:),sgR(:,:),selL(:,:),selR(:,:),selLtmp(:,:)
double complex,allocatable :: gaL(:,:),gaR(:,:)
double complex :: Trans
double precision,allocatable :: mass(:)
integer :: i,j,nL,nR
double complex :: test(2,2)
integer :: intest
double precision :: sum1,sum2
integer:: tag
double precision,allocatable :: Tw(:),omega(:)
tag=1

if (save_tkw) open(101,file="Tkw.txt",recl=1000)
call input_size()
nL=MC(1);nR=MC(N_C)
allocate(sgL(ML0,ML0),sgR(MR0,MR0))
allocate(selL(nL,nL),selLtmp(nL,nL),selR(nR,nR))
allocate(gaL(nL,nL),gaR(nR,nR))
call get_material_para()
call allocate_Force_Constant()
call read_Force_Constant()
call read_Running_Parameters()
call get_center_size
allocate(mass(N_kC))
allocate(Tw(ceiling((wmax-wmin)/dw)+1),omega(ceiling((wmax-wmin)/dw)+1))

do i=1,N_kC;mass(i)=1d0;enddo

!call testroutine(kk,ww,eta,sgr)

do ww=wmin,wmax,dw
omega(tag)=ww*3634.872d0/sqrt(ma) !omega in units of cm^-1, as light velocity
sum1=0d0
do kx=kmin(1),kmax(1),dk(1)
do ky=kmin(2),kmax(2),dk(2)
kk=(/kx,ky/)
call transform_to_kspace(kk)
call sfg_solve(sgR,dcmplx(ww,eta),tkR00,tkR11,tkR01)
call left_lead(ww,eta,ML,ML0,tkL11,tkL01,tkL00,tkL01,sgL,layer)
call self_energy(layer,sgL,sgR,tVLC,tVCR,selL,selR,gaL,gaR)
call transmission(ww,eta,N_kC,N_band,mass,selL,selR,gaL,gaR,Trans)

sum1=sum1+real(Trans)*dk(1)
if (save_tkw) write(101,*)kx,ww,Real(Trans) !Transmission per unit cell
enddo !kx
enddo !ky
if (save_tkw) write(101,*)
Tw(tag)=sum1
tag=tag+1
write(*,'(A9,F6.1,A1)')"Progress:", ww/(wmax-wmin)*100,"%"
enddo !ww

open(103,file='Tw.txt',recl=1000)
do i=1,tag-1
write(103,*)i,omega(i),Tw(i)
enddo
if (save_tkw) close(101)

contains

subroutine testroutine(kk,ww,eta,sgr)
use sizes
use force_constants
implicit none
double complex :: sgr(:,:)
double precision :: kk(2),ww,eta
kk=(/0.31d0,0d0/)
ww=0.001d0
call transform_to_kspace(kk)
write(*,*)tkL11
call sfg_solve(sgR,dcmplx(ww,eta),tkR00,tkR11,tkR01)
call left_lead(ww,eta,ML,ML0,tkL11,tkL01,tkL00,tkL01,sgL,layer)
call self_energy(layer,sgL,sgR,tVLC,tVCR,selL,selR,gaL,gaR)
call transmission(ww,eta,N_kC,N_band,mass,selL,selR,gaL,gaR,Trans)

end subroutine

end program













