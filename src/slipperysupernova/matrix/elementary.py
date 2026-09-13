# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.14.0",
# ]
# ///
"""
Matrix module containing elementary matrix row operations
"""
import torch

def rowswap(a: torch.tensor, sr: int, tr: int):
    """
    swaps two rows in matrix a

    Args:
        a (tensor): a matrix
        sr (int): index of source row
        tr (int): index of target row

    Returns:
        x (tensor): complete matrix

    Raises:
        Index error: if either index is not a correct row

    Examples:
        >>>rowswap([[1,2],[3,5]],0,1)
        [[3,5],[1,2]]
    """
    if sr < 0 | sr> len(a):
        raise IndexError("Index out of range for the source index")
    if tr < 0 | tr > len(a):
        raise IndexError("Index out of range for the target index")
    x = a.detach().clone()
    x[tr] = a[sr]
    x[sr] = a[tr]
    return x

def rowscale(a: torch.tensor, sr: int, sf):
    """
    Scales a row in matrix a by sf

    Args:
        a (tensor): a matrix
        sr (int): index of source row
        sf (number): scaling factor

    Returns:
        x (tensor): complete matrix

    Raises:
        Index error: if the index is not a correct row

    Examples:
        >>>rowscale([[1,2],[3,5]],0,8)
        [[8,16],[3,5]]
    """
    if sr < 0 | sr> len(a):
        raise IndexError("Index out of range for the source index")
    x = a.detach().clone()
    x[sr] = sf*a[sr]
    return x

def rowreplacement(a: torch.tensor, sr: int, tr: int, sf1, sf2):
    """
    Performs sf1*source row + sf2* target row into target row

    Args:
        a (tensor): a matrix
        sr (int): index of source row
        tr (int): index of target row
        sf (number): scaling factor

    Returns:
        x (tensor): complete matrix

    Raises:
        Index error: if either index is not a correct row

    Examples:
        >>>rowreplace([[1,2],[3,5]],0,1,-8,1)
        [[1,2],[-5,-11]]
    """
    if sr < 0 | sr> len(a):
        raise IndexError("Index out of range for the source index")
    if tr < 0 | tr > len(a):
            raise IndexError("Index out of range for the target index")
    x = a.detach().clone()
    x = rowscale(x,sr,sf1)
    x = rowscale(x,tr,sf2)
    x[tr] = x[sr] + x[tr]
    x[sr] = a[sr]
    return x

def rref(a: torch.tensor):
    """
    Turns matrix a into reduced row echelon form

    Args:
        a (tensor): a matrix

    Returns:
        x (tensor): complete matrix
    """
    x = a.detach().clone()
    lead = 0
    row_count = len(x)
    col_count = len(x[0])
    
    for r in range(row_count):
        if col_count <= lead:
            return x
        
        # Step 1: Find a row with a non-zero entry in the current column
        i = r
        while x[i][lead] == 0:
            i += 1
            if row_count == i:
                i = r
                lead += 1
                if col_count == lead:
                    return x
                    
        # Step 2: Swap the current row with the non-zero pivot row
        x = rowswap(x,i,r)
        
        # Step 3: Scale the pivot row so the leading entry becomes 1
        pivot = x[r][lead]
        if pivot != 0:
            x = rowscale(x,r,1/pivot)
            
        # Step 4: Eliminate all other entries (above and below) in this column
        for i in range(row_count):
            if i != r:
                multiplier = x[i][lead]
                x = rowreplacement(x,r,i,-multiplier,1)
                
        lead += 1
        
    return x


if __name__ == '__main__':
    a = torch.tensor([[1.0,3.0,0.0,0.0,3.0],[1.0,0.0,1.0,0.0,9.0],[0.0,0.0,0.0,1.0,-4.0]])
    x = rowswap(a,0,1)
    print(x)
    x =rowscale(a,0,1/3)
    print(x)
    x = rowreplacement(a,0,1,5,-6)
    print(x)
    x = rref(a)
    print(x)
    print(a)