
subroutine Transform_R_to_K(N_cut,R,kk,K)
use material_para
integer :: N_cut(2) !cut-off energy for the matrix in real space
double precision :: kk(2) !The input of k value, kk(1)=kx,kk(2)=ky
double precision :: R(:,:,:,:) !Matrix in real space, first two index: nx, ny, last two index: b,b'
double complex :: K(:,:) !two index represents b and b'

double precision ::vR(2),vK(2)
double complex :: tmp,summ1
integer :: nx,ny,size1,size2,nxx,nyy
integer :: i,j

size1=size(R,dim=3)
size2=size(R,dim=4)

do i=1,size1
do j=1,size2
summ1=0;
do nx=-N_cut(1),N_cut(1)
do ny=-N_cut(2),N_cut(2)
nxx=nx+N_cut(1)+1
nyy=ny+N_cut(2)+1

vR=nx*a1+ny*a2
vK=kk(1)*b1+kk(2)*b2
tmp=-dcmplx(0d0,1d0)*dot_product(vR,vK)
summ1=summ1+R(nxx,nyy,i,j)*exp(tmp)
!write(*,*)i,j,nx,ny,R(nx,ny,i,j)

enddo
enddo
K(i,j)=summ1
enddo
enddo

end subroutine
