subroutine Transmission(ww,eta,N_kC,N_band,mass,selL,selR,gaL,gaR,Trans)
implicit none
interface
function invsG(i,j,N,w,eta,ma,selL,selR)
integer :: i,j,N
double precision :: ma
double complex ::selL(:,:),selR(:,:)
double precision :: w,eta
double complex :: invsG
end function

subroutine solver(N,K,AB,NRHS,B)
Implicit none
integer :: N,K,NRHS
double complex ::AB(:,:),B(:,:)
end subroutine


end interface

double precision :: ww,eta
integer :: N_kC !number of degrees of freedom in the center
integer :: N_band !number of bands of the wI-K-\Sigma
double precision :: mass(:)
double complex :: selL(:,:),selR(:,:),gaL(:,:),gaR(:,:)
double complex :: Trans

!local variables
double complex,allocatable :: Grr(:,:),Gad(:,:)
double complex,allocatable :: AB(:,:)
double complex,allocatable :: Tmatrix(:,:), BL(:,:)
integer :: i,j
integer :: NLC,NRC,K,N
K=N_band
N=N_kC
NLC=size(selL,dim=1)
NRC=size(selR,dim=1)
allocate(Grr(NRC,NLC),Gad(NLC,NRC),Tmatrix(NRC,NRC))
allocate(AB(3*K+1,N),BL(N,NLC))
do i=K+1,3*K+1
 do j=1,N
 if(i+j-2*K-1.ge.1 .and. i+j-2*K-1 .le.N) then
   AB(i,j)=invsG(i+j-2*K-1,j,N,ww,eta,mass(j),selL,selR)
 endif
enddo
enddo
BL=0;
do i=1,NLC;BL(i,i)=1d0;enddo
call solver(N,K,AB,NLC,BL)
do i=1,NRC
Grr(i,:)=BL(N-NRC+i,:)!reduced Gr
enddo
Gad=Transpose(conjg(Grr))
Tmatrix=matmul(Grr,matmul(GaL,matmul(Gad,gaR)))

!do i=1,NLC
!do j=1,NLC
!if (Tmatrix(i,j).ne.0) write(*,*)i,j,Tmatrix(i,j)
!enddo
!enddo





Trans=0;
do i=1,NRC
Trans=Trans+Tmatrix(i,i)
enddo



end subroutine

function invsG(i,j,N,w,eta,ma,selL,selR)
use force_constants
implicit none

integer :: i,j,N
double precision :: ma
double complex ::selL(:,:),selR(:,:)
double precision :: w,eta
double complex :: invsG
integer :: NLC,NRC
double complex :: ww,tmp1,tmp2
NLC=size(selL,dim=1)
NRC=size(selR,dim=1)
ww=(cmplx(w,eta))**2;
if(i.eq.j) then
tmp1=ww*ma-KCC(i,j)
else
 tmp1=-KCC(i,j)
endif
if((i.ge.N-NRC+1).and.(j.ge.N-NRC+1)) tmp1=tmp1-selR(i+NRC-N,j+NRC-N)!right-hand side, its a shifting of index
if((i.le.NLC).and.(j.le.NLC))tmp1=tmp1-selL(i,j)!left-and side, no need shift after flip the index
invsG=tmp1
end function


subroutine solver(N,K,AB,NRHS,B)
!This subroutine is a band matrix solver for square matrix
!N: dimension of A as NxN
!K: number of bands. K=1:tridiagonal, k=2:pentadiagonal
!AB: The matrix in band storage.AB(2K+1+i-j,j)=A(i,j). Dimension of AB:(3K+1),the rows 1 to K need not be set.
!NRHS: The number of columns of B
!The right handside array
Implicit none
integer :: N,K,NRHS
double complex ::AB(:,:),B(:,:)

integer :: KL,KU,LDAB,LDB,INFO
integer,allocatable::IPIV(:)
allocate(IPIV(N))
KL=K;KU=K
LDAB=3*K+1
LDB=N
call ZGBSV(N,KL,KU,NRHS,AB,LDAB, IPIV,B,LDB,INFO)
end subroutine 

