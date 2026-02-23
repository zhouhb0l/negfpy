subroutine left_lead(w,eta,NX,NX0,k11,k01,k00,k12,gr,layer)
implicit none
interface
subroutine flip(A)
double complex :: A(:,:)
end subroutine
subroutine block_flip(A,layer)
double complex :: A(:,:)
integer :: layer(:)
end subroutine

subroutine sfg_solve(GF,omega,h00,h11,h01)
implicit none
!calculate the surface green's function of a quasi-1d semi-infinite periodic
!system using the generalized eigen value method
!
!The arguments are the same as those of Wang Jian's implementation of
!recursive method:
!
!h00 is ((\omega+i\eta)^2-H00) for phonon problem, 
!while it is ((\varepsilon+i\eta)-H00) for electron problem.
!the same with H11
double complex, dimension(:,:), intent(in) :: H00,H01,H11
double complex, dimension(:,:), intent(out) :: GF 
double complex, intent(in) :: omega
!
end subroutine

end interface
double precision :: w,eta
integer :: NX,NX0
double complex :: k11(NX,NX),k01(NX0,NX),k00(NX0,NX0),k12(NX,NX)
double complex :: gr(NX0,NX0)
integer :: layer(:)

!call flip(k00)
!call flip(k01)
!call flip(k11)
!call flip(k12)
!call block_flip(k00,layer)
!call block_flip(k01,layer)
!call block_flip(k11,layer)
!call block_flip(k12,layer)
!call cal_sgf(w,eta,NX,NX0,k11,k01,k00,k12,gr)
!call block_flip(gr,layer)
!call flip(gr)


call sfg_solve(gR,dcmplx(w,eta),k00,k11,k01)



end subroutine



subroutine self_energy(layer,sgL,sgR,VLC,VCR,selL,selR,gaL,gaR)
! the input parameter "layer" is to store  the number of degree of freedom at each layer of the center
! We need this information because the index of the layer need to be flipped during the calculation of left lead self-energy
! But the index of degree of freedom inside each layer need not to be flipped.

implicit none
interface
subroutine flip(A,layer)
double complex :: A(:,:)
integer :: layer(:)
end subroutine
end interface

double complex :: sgL(:,:),sgR(:,:),VLC(:,:),VCR(:,:)
double complex :: selL(:,:),selR(:,:),gaL(:,:),gaR(:,:)
integer :: layer(:)
integer :: nL,nR,i,j

nL=size(VLC,dim=2)
nR=size(VCR,dim=1)
selL=matmul(conjg(transpose(VLC)),matmul(sgL,VLC))
selR=matmul(VCR,matmul(sgR,Conjg(Transpose(VCR))))

!do i=1,nL;do j=1,nL;
!gaL(i,j)=-2d0*aimag(selL(i,j))
!enddo;enddo
!do i=1,nR;do j=1,nR;
!gaR(i,j)=-2d0*aimag(selR(i,j))
!enddo;enddo

gaL=dcmplx(0d0,1d0)*(selL-conjg(transpose(selL)))
gaR=dcmplx(0d0,1d0)*(selR-conjg(transpose(selR)))


end subroutine

subroutine block_flip(A,layer)
double complex :: A(:,:)
integer :: layer(:)
integer :: N_tot,L_tot
integer :: tmp1,tmp2,l,k
integer,allocatable :: cum(:),cum2(:)
double complex,allocatable :: AB(:,:)
N_tot=size(A,dim=1)
L_tot=size(layer) !N_tot=sum of layer(i)
allocate(cum(L_tot+1),cum2(L_tot+1))
allocate(AB(N_tot,N_tot))
tmp1=0;tmp2=N_tot;
do l=1,l_tot
cum(l)=tmp1
cum2(l)=tmp2
tmp1=tmp1+layer(l)
tmp2=tmp2-layer(l)
enddo
cum(l_tot+1)=N_tot;cum2(l_tot+1)=0

do l=1,l_tot
do k=1,l_tot
AB(cum2(l+1)+1:cum2(l),cum2(k+1)+1:cum2(k))=A(cum(l)+1:cum(l+1),cum(k)+1:cum(k+1))
enddo
enddo
A=AB
end subroutine

subroutine flip(A)
double complex :: A(:,:)
!local
double complex,allocatable::B(:,:)
integer :: l1,l2,i1,i2
l1=size(A,dim=1)
l2=size(A,dim=2)
allocate(B(l1,l2))
do i1=1,l1
do i2=1,l2
B(i1,i2)=A(l1-i1+1,l2-i2+1)
enddo
enddo
A=B
end subroutine




