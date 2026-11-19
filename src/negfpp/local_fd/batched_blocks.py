"""Frequency-batched exact block elimination; no global matrix or pair table.

Batch size is a fixed frequency tile, independent of device length. Each block
has shape (nfrequency, nrow, ncolumn); the final spatial block may be smaller.
This is the same augmented triangular algebra independently tested by the
scalar recursive_blocks implementation. No source or operator is repaired.
"""
from dataclasses import dataclass
import numpy as np


def adjoint(a):return a.conj().swapaxes(-1,-2)


@dataclass
class Bands:
    diagonal:list
    upper:list
    lower:list

    @property
    def nbytes(self):return sum(a.nbytes for a in self.diagonal+self.upper+self.lower)


class BatchedFactor:
    def __init__(self,diagonal,upper,lower):
        self.a=Bands([np.array(a,complex,copy=True) for a in diagonal],
                     [np.array(a,complex,copy=True) for a in upper],
                     [np.array(a,complex,copy=True) for a in lower])
        self.sizes=tuple(a.shape[-1] for a in self.a.diagonal)
        if not self.sizes:raise ValueError('Empty block chain')
        self.batch=self.a.diagonal[0].shape[0];n=len(self.sizes)
        if len(upper)!=n-1 or len(lower)!=n-1:raise ValueError('Wrong block count')
        for i,a in enumerate(self.a.diagonal):
            if a.shape!=(self.batch,self.sizes[i],self.sizes[i]):raise ValueError('Invalid diagonal shape')
        for i,(u,l) in enumerate(zip(self.a.upper,self.a.lower)):
            if u.shape!=(self.batch,self.sizes[i],self.sizes[i+1]) or l.shape!=(self.batch,self.sizes[i+1],self.sizes[i]):raise ValueError('Invalid coupling shape')
        if not all(np.isfinite(a).all() for a in self.a.diagonal+self.a.upper+self.a.lower):raise ValueError('Nonfinite operator')
        self.left=[];self.u=[];self.l=[];self.pivot_residual=0.;self.source_calls=0
        self.compacted=False
        self.full_source_calls=0;self.compact_source_calls=0
        self.full_source_block_products_recorded_total=0
        self.compact_source_block_products_recorded_total=0
        self.compact_qualification_source_calls=0
        self.compact_qualification_source_block_products_recorded_total=0
        for i,a in enumerate(self.a.diagonal):
            pivot=a if i==0 else a-self.l[i-1]@self.a.upper[i-1]
            identity=np.broadcast_to(np.eye(self.sizes[i]),pivot.shape)
            g=np.linalg.solve(pivot,identity)
            residue=np.linalg.norm(pivot@g-identity,axis=(-2,-1))/np.sqrt(self.sizes[i])
            self.pivot_residual=max(self.pivot_residual,float(np.max(residue)))
            self.left.append(g)
            if i<n-1:
                self.u.append(g@self.a.upper[i]);self.l.append(self.a.lower[i]@g)
        d=[None]*n;u=[None]*(n-1);l=[None]*(n-1);d[-1]=self.left[-1]
        for i in range(n-2,-1,-1):
            u[i]=-self.u[i]@d[i+1];l[i]=-d[i+1]@self.l[i]
            d[i]=self.left[i]-u[i]@self.l[i]
        self.green=Bands(d,u,l);self.diagonal_residual=0.
        for i in range(n):
            r=self.a.diagonal[i]@d[i]-np.eye(self.sizes[i])
            if i:r+=self.a.lower[i-1]@u[i-1]
            if i<n-1:r+=self.a.upper[i]@l[i]
            self.diagonal_residual=max(self.diagonal_residual,float(np.max(np.linalg.norm(r,axis=(-2,-1))/np.sqrt(self.sizes[i]))))
        if self.pivot_residual>1e-8 or self.diagonal_residual>1e-8:raise ValueError('Recursive inverse residual failed original raw tolerance')

    def compact(self):
        """Release redundant blocks after their unchanged raw qualification.

        Retain only left g, gU, raw lower A blocks and full diagonal G blocks.
        Missing transfers/adjacent G blocks are recomputed exactly as needed.
        Compaction changes cache representation, not the operator or source.
        """
        if self.compacted:return self
        self.raw_lower=self.a.lower
        self.green_diagonal=self.green.diagonal
        self.a=None
        self.l=None
        self.green=None
        self.compacted=True
        return self

    def selected_block(self,row,column):
        """Exact diagonal/adjacent G block, including after compaction."""
        n=len(self.sizes)
        if (not isinstance(row,(int,np.integer)) or not isinstance(column,(int,np.integer))
                or row<0 or column<0 or row>=n or column>=n):raise IndexError('Selected block index outside chain')
        if abs(row-column)>1:raise IndexError('Only diagonal and adjacent blocks are selected')
        if not self.compacted:
            if row==column:return self.green.diagonal[row]
            return self.green.upper[row] if column==row+1 else self.green.lower[column]
        if row==column:return self.green_diagonal[row]
        if column==row+1:return -self.u[row]@self.green_diagonal[column]
        return -self.green_diagonal[row]@(self.raw_lower[column]@self.left[column])

    def correlate(self,s,*,qualification=False):
        if type(qualification) is not bool:raise ValueError('qualification must be boolean')
        n=len(self.sizes)
        if not isinstance(s,Bands) or len(s.diagonal)!=n or len(s.upper)!=n-1 or len(s.lower)!=n-1:raise ValueError('Invalid source chain')
        expected=[(self.batch,b,b) for b in self.sizes]
        expected += [(self.batch,self.sizes[i],self.sizes[i+1]) for i in range(n-1)]
        expected += [(self.batch,self.sizes[i+1],self.sizes[i]) for i in range(n-1)]
        for block,shape in zip(s.diagonal+s.upper+s.lower,expected):
            if not isinstance(block,np.ndarray) or block.shape!=shape or not np.isfinite(block).all():raise ValueError('Invalid source block')
        # Exact independent source lower blocks are used; no conjugate copying.
        for d in s.diagonal:
            if np.max(np.abs(d-adjoint(d)))>1e-12+1e-10*np.max(np.abs(d)):raise ValueError('Non-Hermitian source')
        for u,l in zip(s.upper,s.lower):
            if np.max(np.abs(u-adjoint(l)))>1e-12+1e-10*max(np.max(np.abs(u)),np.max(np.abs(l))):raise ValueError('Source coupling adjoints failed')
        q=[]
        lower=self.raw_lower if self.compacted else self.a.lower
        diagonal_green=self.green_diagonal if self.compacted else self.green.diagonal
        for i,source in enumerate(s.diagonal):
            if i:
                tr=lower[i-1]@self.left[i-1] if self.compacted else self.l[i-1]
                source=source-tr@s.upper[i-1]-s.lower[i-1]@adjoint(tr)+lower[i-1]@q[i-1]@adjoint(lower[i-1])
            q.append(self.left[i]@source@adjoint(self.left[i]))
        d=[None]*n;u=[None]*(n-1);l=[None]*(n-1);d[-1]=q[-1]
        for i in range(n-2,-1,-1):
            b=q[i]@adjoint(lower[i])-self.left[i]@s.upper[i]
            e=lower[i]@q[i]-s.lower[i]@adjoint(self.left[i])
            u[i]=-self.u[i]@d[i+1]-b@adjoint(diagonal_green[i+1])
            l[i]=-diagonal_green[i+1]@e-d[i+1]@adjoint(self.u[i])
            green_upper=-self.u[i]@diagonal_green[i+1] if self.compacted else self.green.upper[i]
            d[i]=q[i]-green_upper@e-u[i]@adjoint(self.u[i])
        if not all(np.isfinite(a).all() for a in d+u+l):raise ValueError('Nonfinite correlation')
        self.source_calls+=1
        products=self.batch*((18 if self.compacted else 16)*n-(16 if self.compacted else 14))
        if self.compacted:
            self.compact_source_calls+=1
            self.compact_source_block_products_recorded_total+=products
            if qualification:
                self.compact_qualification_source_calls+=1
                self.compact_qualification_source_block_products_recorded_total+=products
        else:
            self.full_source_calls+=1
            self.full_source_block_products_recorded_total+=products
        return Bands(d,u,l)

    @property
    def nbytes(self):
        if self.compacted:arrays=self.left+self.u+self.raw_lower+self.green_diagonal
        else:arrays=self.a.diagonal+self.a.upper+self.a.lower+self.left+self.u+self.l+self.green.diagonal+self.green.upper+self.green.lower
        return sum(a.nbytes for a in {id(a):a for a in arrays}.values())

    def stats(self):
        return dict(spatial_blocks=len(self.sizes),frequency_points=self.batch,
                    maximum_solve_dimension=max(self.sizes),factor_block_solves=self.batch*len(self.sizes),
                    factor_cubic_work=self.batch*sum(b**3 for b in self.sizes),
                    compacted=self.compacted,
                    source_calls=self.source_calls,
                    full_source_calls=self.full_source_calls,compact_source_calls=self.compact_source_calls,
                    compact_qualification_source_calls=self.compact_qualification_source_calls,
                    compact_action_source_calls=self.compact_source_calls-self.compact_qualification_source_calls,
                    full_source_block_products_recorded_total=self.full_source_block_products_recorded_total,
                    compact_source_block_products_recorded_total=self.compact_source_block_products_recorded_total,
                    compact_qualification_source_block_products_recorded_total=self.compact_qualification_source_block_products_recorded_total,
                    compact_action_source_block_products_recorded_total=self.compact_source_block_products_recorded_total-self.compact_qualification_source_block_products_recorded_total,
                    source_block_products_recorded_total=self.full_source_block_products_recorded_total+self.compact_source_block_products_recorded_total,
                    source_block_products_per_call=self.batch*((18 if self.compacted else 16)*len(self.sizes)-(16 if self.compacted else 14)),
                    pivot_residual=self.pivot_residual,selected_diagonal_residual=self.diagonal_residual,
                    stored_numeric_bytes=self.nbytes,global_dense_matrix_created=False)
