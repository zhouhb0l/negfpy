
Module force_constants

implicit none

type :: matrix
 double precision, allocatable :: mat(:,:,:,:)
end type matrix
type :: zmatrix
 double complex, allocatable :: zmat(:,:)
end type zmatrix

!force constant in real space
type(matrix), allocatable :: kC_dia(:)
type(matrix), allocatable :: kC_off(:)
double precision, allocatable :: kL11(:,:,:,:),kL01(:,:,:,:),kL00(:,:,:,:)
double precision, allocatable :: kR11(:,:,:,:),kR01(:,:,:,:),kR00(:,:,:,:)
double precision, allocatable :: VLC(:,:,:,:),VCR(:,:,:,:)

!force Constant in K space
double complex, allocatable :: tkL11(:,:),tkL01(:,:),tkL00(:,:)
double complex, allocatable :: tkR11(:,:),tkR01(:,:),tkR00(:,:)
double complex, allocatable :: tVLC(:,:),tVCR(:,:)
type(zmatrix), allocatable :: tkC_dia(:)
type(zmatrix), allocatable :: tkC_off(:)

!sizes of the center
integer :: N_kC  !total degrees of freedom in the center
integer :: N_band ! total number of bands of the center force constants
integer,allocatable :: layer(:) !number of degrees of freedoms in the center coupled to left lead whose index need to be flipped 
!in the self-energy calculation

contains

subroutine get_center_size
use sizes
integer :: i,j,tmp1,tmp2
tmp1=0;
do i=1,N_C;tmp1=tmp1+MC(i);enddo
N_kC=tmp1

If (N_C.eq.1) tmp2=MC(1)
if (N_C.ge.2) then
tmp2=MC(1)+MC(2)
do i=1,N_C-1
if (tmp2.gt.MC(i)+MC(i+1)) tmp2=MC(i)+MC(i+1)
enddo
endif
N_band=tmp2-1 !If N_band=2, it is a pentadiagonal matrix.
end subroutine


subroutine allocate_Force_Constant()
use sizes
implicit none
integer :: i
allocate(kL11(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),ML,ML))
allocate(kL01(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),ML0,ML))
allocate(kL00(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),ML0,ML0))
allocate(kR11(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MR,MR))
allocate(kR01(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MR0,MR))
allocate(kR00(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MR0,MR0))

allocate(kC_dia(N_C),kC_off(N_C-1))
do i=1,N_C
allocate(kC_dia(i)%mat(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MC(i),MC(i)))
enddo
do i=1,N_C-1
allocate(kC_off(i)%mat(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MC(i),MC(i+1)))
enddo
allocate(VLC(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),ML0,MC(1)))
allocate(VCR(-N_cut(1):N_cut(1),-N_cut(2):N_cut(2),MC(N_C),MR0))

allocate(tkL11(ML,ML),tkL01(ML0,ML),tkL00(ML0,ML0))
allocate(tkR11(MR,MR),tkR01(MR0,MR),tkR00(MR0,MR0))

allocate(tkC_dia(N_C),tkC_off(N_C-1))
do i=1,N_C
allocate(tkC_dia(i)%zmat(MC(i),MC(i)))
enddo
do i=1,N_C-1
allocate(tkC_off(i)%zmat(MC(i),MC(i+1)))
enddo
allocate(tVLC(ML0,MC(1)))
allocate(tVCR(MC(N_C),MR0))

end subroutine


subroutine read_force_constant()
use sizes
use material_para

double precision :: k
integer :: i,j,l,m
double precision :: k2,k3

double precision,allocatable :: f(:,:,:,:,:,:,:)
integer :: Na,Nx,Ny,Nz
character :: tmpc
integer :: tmp1
double precision :: tmp2
kR00=0d0
kR01=0d0
kL01=0d0

open(202,file="force/Kmatrix.dat",recl=1000)

read(202,*)tmpc
do l=-N_cut(1),N_cut(1)
do m=-N_cut(2),N_cut(2)
do i=1,MR0
do j=1,MR0
read(202,*)tmp1,tmp1,tmp1,tmp1,tmp2
kR11(l,m,i,j)=tmp2
enddo
enddo
enddo
enddo

read(202,*)tmpc
!write(*,*)tmpc
do l=-N_cut(1),N_cut(1)
do m=-N_cut(2),N_cut(2)
do i=1,MR0
do j=1,MR
read(202,*)tmp1,tmp1,tmp1,tmp1,tmp2
kR01(l,m,i,j)=tmp2
enddo
enddo
enddo
enddo

kR00=kR11;
kC_dia(1)%mat=kR00;
VCR=kR01


kL11=kR11
kL00=kR00
do l=-N_cut(1),N_cut(1)
kL01(l,0,:,:)=Transpose(kR01(-l,0,:,:)) !KL10 is same as kR01
enddo
VLC=kR01

!tmp1=6
!allocate(layer(tmp1))
!do i=1,tmp1;layer(i)=18/tmp1;enddo

end subroutine

subroutine transform_to_kspace(kk)
use sizes
interface
subroutine Transform_R_to_K(N_cut,R,kk,K)
integer :: N_cut(2) !cut-off energy for the matrix in real space
double precision :: kk(2) !The input of k value, kk(1)=kx,kk(2)=ky
double precision :: R(:,:,:,:) !Matrix in real space, first two index: nx, ny, last two index: b,b'
double complex :: K(:,:) !two index represents b and b'
end subroutine
end interface

double precision :: kk(2)
integer :: i

call Transform_R_to_K(N_cut,kL11,kk,tkL11)
call Transform_R_to_K(N_cut,kL01,kk,tkL01)
call Transform_R_to_K(N_cut,kL00,kk,tkL00)
call Transform_R_to_K(N_cut,kR11,kk,tkR11)
call Transform_R_to_K(N_cut,kR01,kk,tkR01)
call Transform_R_to_K(N_cut,kR00,kk,tkR00)
call Transform_R_to_K(N_cut,VLC,kk,tVLC)
call Transform_R_to_K(N_cut,VCR,kk,tVCR)
do i=1,N_C
call Transform_R_to_K(N_cut,kC_dia(i)%mat,kk,tkC_dia(i)%zmat)
enddo
do i=1,N_C-1
call Transform_R_to_K(N_cut,kC_off(i)%mat,kk,tkC_off(i)%zmat)
enddo

end subroutine


function KCC(i,j)
use sizes, Only: N_C,MC
integer :: i,j
double complex :: KCC
integer :: ni,nj,mi,mj !ni,nj determins the block, mi,mj are the index inside block
integer :: x,tmp1
KCC=0

tmp1=0

do x=1,N_C
if (tmp1.lt.i .and. tmp1+MC(x).ge.i) then 
ni=x;mi=i-tmp1
endif

if (tmp1.lt.j .and. tmp1+MC(x).ge.j) then
nj=x;mj=j-tmp1
endif

tmp1=tmp1+MC(x)
enddo


if(ni .eq. nj) then
KCC=tkC_dia(ni)%zmat(mi,mj)
endif

if(ni .eq. nj+1) then
KCC=Conjg(tkC_off(nj)%zmat(mj,mi))
endif

if(ni+1.eq.nj) then
KCC=tkC_off(ni)%zmat(mi,mj)
endif
end function


end module











